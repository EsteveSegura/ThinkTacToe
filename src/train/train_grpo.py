import os
import argparse
import torch
import json
import re
from datetime import datetime
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# Añadir el directorio raíz del proyecto al path
import sys
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
from src.dataset.board_tokenizer import board_to_token_representation

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model using GRPO')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name or path of the base model to use')
    parser.add_argument('--data_files', type=str, required=True,
                      help='Path to the GRPO dataset JSONL file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the trained model')
    parser.add_argument('--logs_dir', type=str, default='./logs',
                      help='Directory to save training logs')
    return parser.parse_args()

class LoggingCallback(TrainerCallback):
    """Callback personalizado para guardar logs de entrenamiento"""
    
    def __init__(self, logs_dir, model_type="grpo"):
        super().__init__()
        self.logs_dir = logs_dir
        self.model_type = model_type
        os.makedirs(logs_dir, exist_ok=True)
        
        # Crear archivo de logs con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(logs_dir, f"{model_type}_training_logs_{timestamp}.txt")
        
        # Escribir header del archivo
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {model_type.upper()} Training Logs ===\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Método llamado cuando se generan logs durante el entrenamiento"""
        if logs is not None:
            # Guardar logs en archivo
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"Step {state.global_step}: {json.dumps(logs, indent=2)}\n")
                f.write("-" * 30 + "\n")
            
            # También imprimir en consola
            print(f"Step {state.global_step}: {logs}")

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

def main():
    args = parse_args()
    
    # Login a Hugging Face
    login(token=os.environ["HF_TOKEN"])

    # Modelo base
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    model.gradient_checkpointing_enable()

    # Cargar dataset
    dataset = load_dataset("json", data_files=args.data_files, split="train")

    # Crear callback para logging
    logging_callback = LoggingCallback(args.logs_dir, "grpo")

    # Configuración GRPO
    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=10,
        learning_rate=5e-6,
        optim="adamw_torch",
        num_generations=4,
        use_liger_loss=True,
        beta=0.0,
        remove_unused_columns=False,
        save_steps=250,
        save_total_limit=1,
        report_to="none",
        fp16=True,
        gradient_checkpointing=True,
        max_prompt_length=256,
        max_completion_length=256,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        scale_rewards=True,
        loss_type="bnpo",
        mask_truncated_completions=False,
        log_completions=False
    )

    # Trainer GRPO
    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_func,
        train_dataset=dataset,
        callbacks=[logging_callback],
    )

    trainer.train()

if __name__ == "__main__":
    main()
