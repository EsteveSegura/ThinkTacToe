#!/usr/bin/env python3
"""
Script de entrenamiento GRPO para Tic-Tac-Toe usando el formato minimax.
Optimizado para movimientos óptimos calculados por algoritmo minimax.
"""

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import json
import os
import sys
import re
from datetime import datetime
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.dataset.board_engine import (
    create_empty_board,
    apply_move,
    get_valid_moves,
    next_player,
    check_winner,
    is_draw
)

# Configurar logging
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_file = f"logs/grpo_minimax_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Crear directorio logs si no existe
os.makedirs("logs", exist_ok=True)

# Clase para capturar la salida y guardarla en archivo
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirigir stdout al archivo
sys.stdout = Logger(log_file)

print(f"=== GRPO Minimax Training Logs ===")
print(f"Started at: {timestamp}")
print("=" * 50)
print()

# Cargar dataset GRPO generado desde minimax
dataset_path = "./datasets/tictactoe_grpo_from_minimax_20250624_175237.jsonl"
print(f"📁 Cargando dataset: {dataset_path}")
dataset = load_dataset("json", data_files=dataset_path, split="train")
print(f"✅ Dataset cargado: {len(dataset)} ejemplos")

def contains_bad_token(text):
    """Verifica si el texto contiene tokens problemáticos o fuera de contexto"""
    bad_patterns = [
        r"<\|endoftext\|>",
        r"<\|system\|>",
        r"<\|user\|>",
        r"<\|assistant\|>",
        r"<\|function\|>",
        r"<\|function_results\|>",
        r"<\|function_calls\|>",
        r"<\|observation\|>",
        r"<\|thought\|>",
        r"<\|action\|>",
        r"<\|action_input\|>",
        r"<\|final_answer\|>",
        r"<document>",
        r"</document>",
        r"<\|document\|>",
        r"</\|document\|>"
    ]
    
    for pattern in bad_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Verificar si contiene múltiples movimientos
    move_count = len(re.findall(r"<\|move\|>", text))
    if move_count > 1:
        return True
    
    return False

def extract_move(text):
    """Extrae el movimiento del texto generado por el modelo"""
    # Corregido para solo permitir coordenadas válidas (0-2)
    match = re.search(r"<\|move\|><\|([0-2])-([0-2])\|><\|end\|>", text)
    return (int(match.group(1)), int(match.group(2))) if match else None

def extract_board_from_prompt(prompt):
    """Extrae el tablero del prompt para evaluar el movimiento"""
    try:
        # Buscar la sección del tablero entre <|board_start|> y <|board_end|>
        board_start = prompt.find("<|board_start|>")
        board_end = prompt.find("<|board_end|>")
        
        if board_start == -1 or board_end == -1:
            return None
            
        board_text = prompt[board_start:board_end]
        
        # Crear tablero vacío
        board = create_empty_board()
        
        # Parsear las líneas del tablero
        lines = board_text.split('\n')[1:-1]  # Excluir las etiquetas de inicio/fin
        
        for i, line in enumerate(lines):
            if i >= 3:  # Solo las primeras 3 líneas
                break
                
            # Buscar patrones como <|0-0|><|X|> <|0-1|><|blank|> <|0-2|><|O|>
            cell_pattern = r'<\|(\d)-(\d)\|><\|([^|]+)\|>'
            matches = re.findall(cell_pattern, line)
            
            for match in matches:
                row, col, value = int(match[0]), int(match[1]), match[2]
                if value == 'X':
                    board[row][col] = 'X'
                elif value == 'O':
                    board[row][col] = 'O'
                # 'blank' se mantiene como None
        
        return board
    except Exception as e:
        print(f"Error parsing board: {e}")
        return None

def get_current_player_from_prompt(prompt):
    """Extrae el jugador actual del prompt - CORREGIDO"""
    # Corregido para buscar <|player|>X o <|player|>O en lugar de <|turn|>
    m = re.search(r'<\|player\|>([XO])', prompt)
    return m.group(1) if m else None

def evaluate_move_quality_minimax(board, move, current_player):
    """Evalúa la calidad del movimiento comparándolo con el movimiento óptimo de minimax"""
    if not board or not move:
        return 0.0
    
    row, col = move
    
    # Verificar si el movimiento es válido
    valid_moves = get_valid_moves(board)
    if move not in valid_moves:
        return -4.0  # Movimiento inválido - penalización más fuerte
    
    # Aplicar el movimiento
    new_board = apply_move(board, current_player, move)
    
    # Verificar si es un movimiento ganador
    if check_winner(new_board) == current_player:
        return 2.0  # Movimiento ganador - recompensa máxima
    
    # Verificar si bloquea un movimiento ganador del oponente
    # CORREGIDO: Verificar en el tablero original, no después de aplicar nuestro movimiento
    opponent = 'O' if current_player == 'X' else 'X'
    for valid_move in valid_moves:
        opponent_board = apply_move(board, opponent, valid_move)
        if check_winner(opponent_board) == opponent:
            # Si el oponente puede ganar con este movimiento, nuestro movimiento debe bloquearlo
            if move == valid_move:
                return 1.5  # Movimiento bloqueador - recompensa alta
    
    # Evaluación estratégica para movimientos no ganadores/no bloqueadores
    score = 0.0
    
    # Recompensar movimientos en el centro (estratégicamente importantes)
    if move == (1, 1):
        score += 0.8
    
    # Recompensar movimientos en esquinas (buenas posiciones estratégicas)
    elif move in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        score += 0.6
    
    # Recompensar movimientos que crean amenazas (dos en línea)
    threat_created = False
    for direction in [(0, 1), (1, 0), (1, 1), (1, -1)]:  # horizontal, vertical, diagonales
        count = 1  # Contar nuestro movimiento
        # Contar en dirección positiva
        for step in range(1, 3):
            new_row, new_col = row + step * direction[0], col + step * direction[1]
            if 0 <= new_row < 3 and 0 <= new_col < 3 and new_board[new_row][new_col] == current_player:
                count += 1
            else:
                break
        # Contar en dirección negativa
        for step in range(1, 3):
            new_row, new_col = row - step * direction[0], col - step * direction[1]
            if 0 <= new_row < 3 and 0 <= new_col < 3 and new_board[new_row][new_col] == current_player:
                count += 1
            else:
                break
        
        if count >= 2:
            threat_created = True
            break
    
    if threat_created:
        score += 0.4
    
    # Recompensar movimientos que bloquean amenazas del oponente
    opponent_threats_blocked = 0
    for direction in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        count = 0
        for step in range(-2, 3):
            new_row, new_col = row + step * direction[0], col + step * direction[1]
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                if board[new_row][new_col] == opponent:
                    count += 1
                elif board[new_row][new_col] == current_player:
                    count = 0  # Reset si encontramos nuestra pieza
                    break
        if count >= 2:
            opponent_threats_blocked += 1
    
    if opponent_threats_blocked > 0:
        score += 0.3 * opponent_threats_blocked
    
    # Recompensa base para movimientos válidos
    base_reward = 0.2
    
    return base_reward + score

def evaluate_format_quality(completion):
    """Evalúa la calidad del formato de la completion"""
    if not completion:
        return -1.0
    
    score = 0.0
    
    # Verificar formato básico del movimiento - CORREGIDO para coordenadas válidas
    if re.fullmatch(r"<\|move\|><\|[0-2]-[0-2]\|><\|end\|>", completion.strip()):
        score += 1.0  # Formato perfecto
    elif re.search(r"<\|move\|><\|[0-2]-[0-2]\|>", completion):
        score += 0.5  # Formato parcialmente correcto
    else:
        score -= 1.0  # Formato incorrecto
    
    # Penalizar texto extra
    if len(completion.strip()) > 50:  # Debería ser muy corto
        score -= 0.5
    
    return score

def reward_func(completions, prompts=None, **kwargs):
    """Función de recompensa optimizada para el formato minimax"""
    rewards = []
    
    for i, completion in enumerate(completions):
        prompt = prompts[i] if prompts else ""
        reward = 0.0
        
        # PENALIZACIÓN FUERTE por texto fuera de contexto
        if contains_bad_token(completion):
            reward = -3.0
            rewards.append(reward)
            continue
        
        # Extraer el movimiento del texto generado
        move = extract_move(completion)
        
        if not move:
            reward = -2.0  # No se puede extraer movimiento
            rewards.append(reward)
            continue
        
        # Extraer el tablero y jugador del prompt
        board = extract_board_from_prompt(prompt)
        current_player = get_current_player_from_prompt(prompt)
        
        if not board or not current_player:
            reward = -1.0  # No se puede extraer información del prompt
            rewards.append(reward)
            continue
        
        # Evaluar la calidad del movimiento
        move_quality = evaluate_move_quality_minimax(board, move, current_player)
        
        # Evaluar la calidad del formato
        format_quality = evaluate_format_quality(completion)
        
        # Combinar recompensas - Ajustado para mantener simetría
        final_reward = move_quality + format_quality * 0.5  # Reducir peso del formato
        rewards.append(final_reward)
    
    # Asegurar que tenemos el mismo número de recompensas que completions
    assert len(rewards) == len(completions), f"Recompensas: {len(rewards)}, Completions: {len(completions)}"
    
    return rewards

# Configuración para entrenamiento GRPO optimizado
training_args = GRPOConfig(
    output_dir="qwen2.5-0.5b-tictactoe-grpo-minimax",
    num_train_epochs=1,
    per_device_train_batch_size=4,  # Reducido aún más para estabilidad
    gradient_accumulation_steps=4,  # Aumentado para mantener batch efectivo
    learning_rate=5e-6,  # Learning rate más conservador
    bf16=True,  # Cambiado de fp16 a bf16 para estabilidad
    gradient_checkpointing=True,
    save_steps=100,
    save_total_limit=2,
    logging_steps=25,
    warmup_steps=20,
    max_completion_length=12,  # Reducido para evitar spam
    temperature=0.7,  # Aumentado para evitar problemas numéricos
    top_p=0.9,
    repetition_penalty=1.0,  # Reducido para evitar problemas
    num_generations=2,  # Reducido para estabilidad
    max_prompt_length=256,  # Prompts más cortos
    remove_unused_columns=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    # Configuraciones adicionales para estabilidad
    max_grad_norm=1.0,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    # Configuraciones GRPO específicas
    loss_type="dr_grpo",
    scale_rewards=False,
    # Configuraciones de generación más conservadoras
)

print(f"🤖 Iniciando entrenamiento GRPO con configuración optimizada...")
print(f"📊 Configuración:")
print(f"   - Learning rate: {training_args.learning_rate}")
print(f"   - Batch size: {training_args.per_device_train_batch_size}")
print(f"   - Epochs: {training_args.num_train_epochs}")
print(f"   - Max completion length: {training_args.max_completion_length}")

# Inicializar trainer (manejo automático del tokenizer como en train_grpo_simple.py)
print(f"📁 Cargando modelo base para GRPO...")
trainer = GRPOTrainer(
    model="qwen2.5-0.5b-tictactoe-sft-nothink-minmax/checkpoint-156"
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)

print("🚀 Comenzando entrenamiento...")
trainer.train()

print("✅ Entrenamiento completado!") 