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
from transformers import AutoTokenizer, TrainerCallback
import torch

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

dataset_path = "./datasets/tictactoe_grpo_from_minimax_20250628_220808.jsonl"
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
    """Extrae el movimiento: admite que la completion empiece justo por la coordenada."""
    # Caso 1: completion contiene todo el patrón completo
    m = re.search(r"<\|move\|><\|([0-2])-([0-2])\|><\|end\|>", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Caso 2: solo la coordenada + sufijos, p.ej. "2-0|><|end|>"
    m = re.search(r"([0-2])-([0-2])\|><\|end\|>", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def extract_board_from_prompt(prompt):
    """Extrae el tablero del prompt para evaluar el movimiento"""
    try:
        # Buscar la sección del tablero entre <|board_start|> y <|board_end|>
        board_start_tag = "<|board_start|>"
        board_end_tag = "<|board_end|>"
        board_start = prompt.find(board_start_tag)
        board_end = prompt.find(board_end_tag)
        
        if board_start == -1 or board_end == -1:
            return None
            
        # Extraer solo el contenido entre los tags (sin incluir los tags)
        board_text = prompt[board_start + len(board_start_tag):board_end]
        
        # Crear tablero vacío
        board = create_empty_board()
        
        # Parsear las líneas del tablero (deben ser exactamente 3)
        lines = [l.strip() for l in board_text.strip().split('\n') if l.strip()]
        
        for i, line in enumerate(lines):
            if i >= 3:
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
    """Extrae el jugador actual del prompt (acepta <|symbol|> o <|player|>)"""
    m = re.search(r'<\|(symbol|player)\|>([XO])', prompt)
    return m.group(2) if m else None

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
    """Evalúa formato uniendo el prefijo estándar si no está presente."""
    if not completion:
        return -1.0

    completion = completion.strip()
    # Asegurar que evaluamos contra la cadena completa
    full_text = completion if completion.startswith('<|move|><|') else ('<|move|><|' + completion)

    if re.fullmatch(r"<\|move\|><\|[0-2]-[0-2]\|><\|end\|>", full_text):
        return 1.0
    if re.search(r"<\|move\|><\|[0-2]-[0-2]\|><\|end\|>", full_text):
        return 0.4
    return -1.0

def reward_func(completions, prompts=None, **kwargs):
    """Función de recompensa optimizada para el formato minimax con logging de depuración"""
    # Contadores para diagnóstico
    if not hasattr(reward_func, "call_id"):
        reward_func.call_id = 0
    reward_func.call_id += 1

    stats = {
        "total": 0,
        "bad_token": 0,
        "no_move": 0,
        "no_prompt_info": 0,
        "invalid_move": 0,
        "ok": 0,
    }

    rewards = []
    
    for i, completion in enumerate(completions):
        prompt = prompts[i] if prompts else ""
        stats["total"] += 1
        reward = 0.0
        
        # PENALIZACIÓN FUERTE por texto fuera de contexto
        if contains_bad_token(completion):
            reward = -3.0
            stats["bad_token"] += 1
            rewards.append(reward)
            continue
        
        # Extraer el movimiento del texto generado
        move = extract_move(completion)
        
        if not move:
            reward = -2.0  # No se puede extraer movimiento
            stats["no_move"] += 1
            rewards.append(reward)
            continue
        
        # Extraer el tablero y jugador del prompt
        board = extract_board_from_prompt(prompt)
        current_player = get_current_player_from_prompt(prompt)
        
        if not board or not current_player:
            reward = -1.0  # No se puede extraer información del prompt
            stats["no_prompt_info"] += 1
            rewards.append(reward)
            continue
        
        # Evaluar la calidad del movimiento
        move_quality = evaluate_move_quality_minimax(board, move, current_player)
        if move_quality == -4.0:
            stats["invalid_move"] += 1
        
        # Evaluar la calidad del formato
        format_quality = evaluate_format_quality(completion)
        
        # Combinar recompensas - Ajustado para mantener simetría
        final_reward = move_quality + format_quality * 0.5  # Reducir peso del formato
        rewards.append(final_reward)
        stats["ok"] += 1
    
    # Asegurar que tenemos el mismo número de recompensas que completions
    assert len(rewards) == len(completions), f"Recompensas: {len(rewards)}, Completions: {len(completions)}"

    # --- Logging de diagnóstico ---
    if reward_func.call_id % 25 == 0:  # Cada 25 llamadas (~ logging_steps)
        print(f"\n[REWARD DEBUG] Llamada #{reward_func.call_id}")
        print("  Total completions:", stats["total"])
        print("  Sin movimiento extraíble:", stats["no_move"])
        print("  Bad token:", stats["bad_token"])
        print("  Sin info prompt:", stats["no_prompt_info"])
        print("  Movimiento inválido:", stats["invalid_move"])
        print("  OK:", stats["ok"])
        if stats["no_move"] > 0:
            # Imprimir ejemplo de completion para inspección
            idx = completions.index(next(c for c in completions if extract_move(c) is None))
            print("  Ejemplo sin movimiento =>", completions[idx][:120])
    
    return rewards

# Configuración para entrenamiento GRPO optimizado
training_args = GRPOConfig(
    output_dir="qwen2.5-0.5b-tictactoe-grpo-minimax",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Reducido aún más para estabilidad
    gradient_accumulation_steps=4,  # Aumentado para mantener batch efectivo
    learning_rate=5e-6,  # Learning rate más conservador
    bf16=True,  # Cambiado de fp16 a bf16 para estabilidad
    gradient_checkpointing=False,
    save_steps=100,
    save_total_limit=2,
    logging_steps=25,
    warmup_steps=100,
    max_completion_length=20,  # Reducido para evitar spam
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
    lr_scheduler_type="constant_with_warmup",
    # Configuraciones GRPO específicas
    loss_type="dr_grpo",
    scale_rewards=True,
    advantage_fn="reward_minus_baseline_ema",
    baseline_beta=0.95,
    log_keys=["reward", "advantage", "policy_loss", "grad_norm"],
)

print(f"🤖 Iniciando entrenamiento GRPO con configuración optimizada...")
print(f"📊 Configuración:")
print(f"   - Learning rate: {training_args.learning_rate}")
print(f"   - Batch size: {training_args.per_device_train_batch_size}")
print(f"   - Epochs: {training_args.num_train_epochs}")
print(f"   - Max completion length: {training_args.max_completion_length}")

# Inicializar trainer (manejo automático del tokenizer como en train_grpo_simple.py)
print(f"📁 Cargando modelo base para GRPO...")

# Cargar tokenizer del modelo base para pruebas de generación
print("🔤 Cargando tokenizer para inferencias de test...")
try:
    tokenizer = AutoTokenizer.from_pretrained("qwen2.5-0.5b-tictactoe-sft-nothink-minmax", trust_remote_code=True)
except Exception as e:
    print(f"⚠️  No se pudo cargar el tokenizer automáticamente: {e}")
    tokenizer = None

class InferenceCallback(TrainerCallback):
    """Realiza inferencias rápidas sobre un pequeño set de prompts cada N steps"""
    def __init__(self, tokenizer, prompts, every_n_steps=100, max_new_tokens=20):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens

    def on_step_end(self, args, state, control, **kwargs):
        if self.tokenizer is None:
            return control
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        # Generar inferencias
        model.eval()
        inputs = self.tokenizer(self.prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens, temperature=0.7, top_p=0.9)
        outputs = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        print("\n===== Inferencias de test (paso", state.global_step, ") =====")
        for prompt, out in zip(self.prompts, outputs):
            first_line_prompt = prompt.replace("\n", " ")[:60]
            print("P:", first_line_prompt)
            print("→", out)
            print("---")
        model.train()
        return control

# Seleccionar algunos prompts de ejemplo del propio dataset
sample_prompts = [dataset[i]["prompt"] for i in range(min(5, len(dataset)))]

trainer = GRPOTrainer(
    model="qwen2.5-0.5b-tictactoe-sft-nothink-minmax/checkpoint-156",
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)

# Añadir callback de inferencia al trainer
trainer.add_callback(InferenceCallback(tokenizer, sample_prompts, every_n_steps=100))

print("🚀 Comenzando entrenamiento...")
trainer.train()

print("✅ Entrenamiento completado!") 