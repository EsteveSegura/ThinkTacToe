from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import sys
import os
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.dataset.board_engine import (
    create_empty_board,
    apply_move,
    print_board,
    get_valid_moves,
    next_player
)
from src.dataset.board_tokenizer import board_to_token_representation

def create_random_board() -> list:
    """
    Crea un tablero aleatorio con algunos movimientos ya realizados.
    """
    board = create_empty_board()
    num_moves = random.randint(2, 6)  # Entre 2 y 6 movimientos iniciales
    
    for _ in range(num_moves):
        valid_moves = get_valid_moves(board)
        if not valid_moves:
            break
        move = random.choice(valid_moves)
        player = next_player(board)
        board = apply_move(board, player, move)
    
    return board

def parse_model_move(model_output: str) -> tuple:
    """
    Extrae las coordenadas del movimiento del modelo.
    Ejemplo: <|move|><|2-2|><|end|> -> (2, 2)
    """
    try:
        # Extraer la parte entre <| y |>
        move_part = model_output.split("<|move|><|")[1].split("|><|end|>")[0]
        row, col = map(int, move_part.split("-"))
        return (row, col)
    except:
        return None

def visualize_game_state(initial_board: list, model_move: tuple = None):
    """
    Visualiza el estado inicial del tablero y el movimiento del modelo.
    """
    print("\nEstado inicial del tablero:")
    print_board(initial_board)
    
    if model_move:
        row, col = model_move
        print(f"\nMovimiento del modelo: ({row}, {col})")
        
        # Aplicar el movimiento
        final_board = apply_move(initial_board, 'Ô', model_move)
        print("\nEstado final del tablero:")
        print_board(final_board)

def infer(prompt: str, max_new_tokens: int = 300):
    """
    Realiza la inferencia con el modelo y retorna el movimiento.
    """
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

if __name__ == "__main__":
    # Configuración del modelo
    model_name = "./qwen2.5-1.5b-tictactoe/checkpoint-6750"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

    # Generar y procesar tableros aleatorios
    num_games = 5  # Número de juegos a simular
    
    for game in range(num_games):
        print(f"\n{'='*50}")
        print(f"Juego {game + 1}/{num_games}")
        print(f"{'='*50}")
        
        # Crear tablero aleatorio
        board = create_random_board()
        
        # Convertir a formato de prompt
        prompt = board_to_token_representation(board)
        
        # Obtener respuesta del modelo
        model_output = infer(prompt)
        print("\nRespuesta del modelo:", model_output)
        
        # Extraer y visualizar el movimiento
        move = parse_model_move(model_output)
        if move:
            visualize_game_state(board, move)
        else:
            print("\nError al parsear el movimiento del modelo")
            visualize_game_state(board) 