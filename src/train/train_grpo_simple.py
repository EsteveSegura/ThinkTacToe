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

def reward_func(completions, prompts=None, **kwargs):
    """Función de recompensa que evalúa la calidad de los movimientos"""
    rewards = []
    
    for i, completion in enumerate(completions):
        prompt = prompts[i] if prompts else ""
        
        # Extraer el movimiento del texto generado
        move = extract_move(completion)
        
        if not move:
            rewards.append(-1.0)  # Penalizar si no se puede extraer el movimiento
            continue
        
        # Extraer el tablero del prompt
        board = extract_board_from_prompt(prompt)
        
        # Evaluar la calidad del movimiento
        reward = evaluate_move_quality(board, move)
        rewards.append(reward)
    
    return rewards

# Configuración para entrenamiento completo en H100
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
    max_prompt_length=128,
    max_completion_length=128,
)

trainer = GRPOTrainer(
    model="GiRLaZo/qwen2.5-0.5b-tictactoe-sft-llm",
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
