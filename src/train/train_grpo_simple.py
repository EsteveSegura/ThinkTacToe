# train_grpo_simple.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import json
import os
import sys
import re
from datetime import datetime

# Añadir el directorio raíz del proyecto al path
from pathlib import Path
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

# Configurar logging simple
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_file = f"logs/grpo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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

print(f"=== GRPO Training Logs ===")
print(f"Started at: {timestamp}")
print("=" * 50)
print()

# Cargar dataset local
dataset = load_dataset("json", data_files="./datasets/tictactoe_grpo_llm.jsonl", split="train")

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
        r"<\|final_answer\|>"
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
    match = re.search(r"<\|move\|><\|(\d)-(\d)\|><\|end\|>", text)
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

def evaluate_move_quality(board, move):
    """Evalúa la calidad del movimiento en el contexto del tablero"""
    if not board or not move:
        return 0.0
    
    row, col = move
    
    # Verificar si el movimiento es válido
    valid_moves = get_valid_moves(board)
    if move not in valid_moves:
        return -1.0  # Movimiento inválido
    
    # Aplicar el movimiento
    new_board = apply_move(board, 'X', move)
    
    # Verificar si es un movimiento ganador
    if check_winner(new_board) == 'X':
        return 1.0  # Movimiento ganador
    
    # Verificar si bloquea un movimiento ganador del oponente
    opponent_board = apply_move(board, 'O', move)
    if check_winner(opponent_board) == 'O':
        return 0.8  # Movimiento bloqueador
    
    # Verificar si es un movimiento estratégico (centro, esquinas)
    if move == (1, 1):  # Centro
        return 0.6
    elif move in [(0, 0), (0, 2), (2, 0), (2, 2)]:  # Esquinas
        return 0.4
    else:  # Laterales
        return 0.2

def evaluate_thinking_quality(completion):
    """Evalúa la calidad del pensamiento del modelo"""
    if not completion:
        return 0.0
    
    # Extraer el contenido del pensamiento
    think_match = re.search(r"<player_think>(.*?)</player_think>", completion, re.DOTALL)
    if not think_match:
        return 0.0
    
    thinking = think_match.group(1).strip()
    
    # Criterios de calidad del pensamiento
    score = 0.0
    
    # Longitud mínima (debe ser sustancial)
    if len(thinking) < 50:
        score -= 0.3
    elif len(thinking) > 200:
        score += 0.2
    
    # Debe contener análisis estratégico
    strategic_keywords = [
        'analyze', 'strategy', 'strategic', 'position', 'control', 'threat',
        'win', 'block', 'diagonal', 'row', 'column', 'center', 'corner',
        'opportunity', 'advantage', 'flexibility', 'multiple', 'future'
    ]
    
    keyword_count = sum(1 for keyword in strategic_keywords if keyword.lower() in thinking.lower())
    score += min(keyword_count * 0.1, 0.5)  # Máximo 0.5 por keywords
    
    # Debe mencionar el movimiento específico
    if re.search(r'\(\d,\d\)', thinking):
        score += 0.2
    
    # Debe explicar el razonamiento
    if any(word in thinking.lower() for word in ['because', 'since', 'therefore', 'thus', 'so']):
        score += 0.2
    
    return max(score, -0.5)  # No penalizar demasiado

def reward_func(completions, prompts=None, **kwargs):
    """Función de recompensa que evalúa la calidad de los movimientos y el pensamiento"""
    rewards = []
    
    for i, completion in enumerate(completions):
        prompt = prompts[i] if prompts else ""
        reward = 0.0  # Recompensa base
        
        # PENALIZACIÓN FUERTE por texto fuera de contexto
        if contains_bad_token(completion):
            reward = -2.0
            rewards.append(reward)
            continue
        
        # Verificar formato básico
        think_count = completion.count("</player_think>")
        if think_count == 0:
            reward = -1.0  # Falta pensamiento
            rewards.append(reward)
            continue
        elif think_count > 1:
            reward -= 0.5  # Demasiados cierres
        
        # Verificar que después de </player_think> solo esté el movimiento
        if "</player_think>" in completion:
            after_think = completion.split("</player_think>")[-1].strip()
            if not re.fullmatch(r"<\|move\|><\|\d-\d\|><\|end\|>", after_think):
                reward -= 0.5  # Formato incorrecto después del pensamiento
        
        # Extraer el movimiento del texto generado
        move = extract_move(completion)
        
        if not move:
            reward = -1.0  # No se puede extraer movimiento
            rewards.append(reward)
            continue
        
        # Extraer el tablero del prompt
        board = extract_board_from_prompt(prompt)
        
        # Evaluar la calidad del movimiento
        move_quality = evaluate_move_quality(board, move)
        
        # Evaluar la calidad del pensamiento
        thinking_quality = evaluate_thinking_quality(completion)
        
        # BONUS por formato perfecto
        format_bonus = 0.1
        
        final_reward = move_quality + thinking_quality + format_bonus + reward
        rewards.append(final_reward)
    
    # Asegurar que tenemos el mismo número de recompensas que completions
    assert len(rewards) == len(completions), f"Recompensas: {len(rewards)}, Completions: {len(completions)}"
    
    return rewards

# Configuración para entrenamiento estable en H100
training_args = GRPOConfig(
    output_dir="qwen2.5-0.5b-tictactoe-grpo",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    bf16=True,
    gradient_checkpointing=True,
    save_steps=200,
    save_total_limit=1,
    logging_steps=50,
    warmup_steps=10,
    max_completion_length=256,  # Aumentado para permitir pensamientos más largos
    temperature=0.7,  # Añadido para controlar la creatividad
    top_p=0.9,  # Añadido para diversidad
    repetition_penalty=1.1,  # Añadido para evitar repeticiones
    num_generations=4,  # Número de generaciones por prompt
    max_prompt_length=512,  # Longitud máxima del prompt
    remove_unused_columns=False,  # Importante para GRPO
    dataloader_num_workers=0,  # Evitar problemas de multiprocessing
    dataloader_pin_memory=False,  # Evitar problemas de memoria
)

trainer = GRPOTrainer(
    model="GiRLaZo/qwen2.5-0.5b-tictactoe-sft-llm",
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
    tokenizer=None,  # Usar el tokenizer del modelo
)

trainer.train()
