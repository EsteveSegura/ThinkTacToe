from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
from typing import List, Tuple, Optional
import re
import sys
import os
import time

# Añadir el directorio raíz al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dataset.board_engine import (
    create_empty_board,
    get_valid_moves,
    apply_move,
    check_winner,
    is_draw,
    generate_win_scenario,
    generate_block_scenario,
    generate_neutral_scenario
)
from src.dataset.board_tokenizer import board_to_token_representation

# Inicializar el generador aleatorio con una semilla basada en el tiempo
random.seed(time.time())

# Configuración del modelo
model_name = "./qwen2.5-1.5b-tictactoe/checkpoint-3500"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

def parse_move(move_str: str) -> Tuple[int, int]:
    """Extrae las coordenadas del movimiento del formato tokenizado."""
    match = re.search(r'<\|(\d)-(\d)\|>', move_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    raise ValueError(f"Formato de movimiento inválido: {move_str}")

def print_board(board: List[List[Optional[str]]], title: str = ""):
    """Imprime el tablero en la terminal."""
    if title:
        print(f"\n{title}")
    print("  0 1 2")
    for i, row in enumerate(board):
        print(f"{i} ", end="")
        for cell in row:
            print(f"{cell if cell else ' '} ", end="")
        print()

def infer(prompt: str, max_new_tokens: int = 300) -> str:
    """Realiza la inferencia con el modelo."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    eos_token = "<|end|>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token) if eos_token in tokenizer.get_vocab() else tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if "<|end|>" in output_text:
        output_text = output_text.split("<|end|>")[0] + "<|end|>"

    return output_text

def generate_random_board() -> List[List[Optional[str]]]:
    """Genera un tablero aleatorio con algunos movimientos."""
    board = create_empty_board()
    players = ['X', 'O']
    current_player = 0
    
    # Hacer entre 2 y 4 movimientos aleatorios
    num_moves = random.randint(2, 4)
    for _ in range(num_moves):
        valid_moves = get_valid_moves(board)
        if not valid_moves:
            break
        # Mezclar los movimientos válidos para mayor aleatoriedad
        random.shuffle(valid_moves)
        move = valid_moves[0]
        board = apply_move(board, players[current_player], move)
        current_player = (current_player + 1) % 2
    
    return board

def generate_scenario_board() -> List[List[Optional[str]]]:
    """Genera un tablero con un escenario específico o aleatorio."""
    scenarios = [
        generate_win_scenario,
        generate_block_scenario,
        generate_neutral_scenario,
        generate_random_board
    ]
    # Mezclar los escenarios para mayor aleatoriedad
    random.shuffle(scenarios)
    return scenarios[0]()

def main():
    """Función principal que genera tableros, hace inferencia y visualiza los resultados."""
    while True:
        # Generar tablero con escenario específico o aleatorio
        board = generate_scenario_board()
        
        # Convertir a formato tokenizado
        board_str = board_to_token_representation(board)
        
        # Mostrar tablero original
        print("\n" + "="*50)
        print("Tablero Original:")
        print_board(board)
        
        # Realizar inferencia
        print("\nRealizando inferencia...")
        response = infer(board_str)
        print(f"Respuesta del modelo: {response}")
        
        # Extraer y aplicar el movimiento
        try:
            move = parse_move(response)
            new_board = apply_move(board, 'X', move)
            
            # Mostrar tablero con el movimiento
            print("\nTablero con Movimiento:")
            print_board(new_board)
            
            # Verificar resultado
            winner = check_winner(new_board)
            if winner:
                print(f"\n¡{winner} ha ganado!")
            elif is_draw(new_board):
                print("\n¡Empate!")
                
        except ValueError as e:
            print(f"Error al procesar el movimiento: {e}")
        
        # Preguntar si continuar
        if input("\n¿Continuar? (s/n): ").lower() != 's':
            break

if __name__ == "__main__":
    main() 