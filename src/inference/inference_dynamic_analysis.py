from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.dataset.board_engine import (
    create_empty_board,
    apply_move,
    print_board,
    get_valid_moves,
    next_player,
    check_winner
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

def evaluate_move(board: list, move: tuple) -> dict:
    """
    Evalúa si el movimiento es válido y bueno.
    Retorna un diccionario con la evaluación.
    """
    if not move:
        return {
            'is_valid': False,
            'is_winning': False,
            'is_blocking': False,
            'is_center': False,
            'is_corner': False,
            'is_edge': False
        }
    
    row, col = move
    
    # Verificar si el movimiento es válido
    valid_moves = get_valid_moves(board)
    is_valid = move in valid_moves
    
    if not is_valid:
        return {
            'is_valid': False,
            'is_winning': False,
            'is_blocking': False,
            'is_center': False,
            'is_corner': False,
            'is_edge': False
        }
    
    # Aplicar el movimiento
    new_board = apply_move(board, 'X', move)
    
    # Verificar si es un movimiento ganador
    is_winning = check_winner(new_board) == 'X'
    
    # Verificar si bloquea un movimiento ganador del oponente
    is_blocking = False
    if not is_winning:
        opponent_board = apply_move(board, 'O', move)
        is_blocking = check_winner(opponent_board) == 'O'
    
    # Verificar la posición del movimiento
    is_center = (row, col) == (1, 1)
    is_corner = (row, col) in [(0, 0), (0, 2), (2, 0), (2, 2)]
    is_edge = not (is_center or is_corner)
    
    return {
        'is_valid': True,
        'is_winning': is_winning,
        'is_blocking': is_blocking,
        'is_center': is_center,
        'is_corner': is_corner,
        'is_edge': is_edge
    }

def infer(prompt: str, max_new_tokens: int = 32):
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

def board_to_string(board: list) -> str:
    """
    Convierte el tablero a una representación de string.
    """
    if not board:
        return "Tablero vacío"
    
    # Convertir cada elemento a string y manejar None
    return '\n'.join([''.join(str(cell) if cell is not None else ' ' for cell in row) for row in board])

if __name__ == "__main__":
    # Configuración del modelo
    model_name = "-"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

    # Generar y procesar tableros aleatorios
    num_games = 5
    results = []
    
    for game in range(num_games):
        print(f"\nProcesando juego {game + 1}/{num_games}")
        
        # Crear tablero aleatorio
        board = create_random_board()
        
        # Convertir a formato de prompt
        prompt = board_to_token_representation(board)
        
        # Obtener respuesta del modelo
        model_output = infer(prompt)
        
        # Extraer el movimiento
        move = parse_model_move(model_output)
        
        # Evaluar el movimiento
        evaluation = evaluate_move(board, move)
        
        # Guardar resultados
        result = {
            'game_id': game + 1,
            'initial_board': board_to_string(board),
            'model_output': model_output,
            'move': str(move) if move else 'None',
            **evaluation
        }
        results.append(result)
        
        # Mostrar progreso cada 50 juegos
        if (game + 1) % 50 == 0:
            print(f"Completados {game + 1} juegos")
    
    # Crear DataFrame y guardar resultados
    df = pd.DataFrame(results)
    
    # Crear directorio de resultados si no existe
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Generar nombre de archivo con timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'model_evaluation_{timestamp}.csv'
    
    # Guardar resultados
    df.to_csv(output_file, index=False)
    
    # Mostrar resumen
    print("\nResumen de la evaluación:")
    print(f"Total de juegos: {len(df)}")
    print(f"Movimientos válidos: {df['is_valid'].sum()} ({df['is_valid'].mean()*100:.1f}%)")
    print(f"Movimientos ganadores: {df['is_winning'].sum()} ({df['is_winning'].mean()*100:.1f}%)")
    print(f"Movimientos bloqueadores: {df['is_blocking'].sum()} ({df['is_blocking'].mean()*100:.1f}%)")
    print(f"\nDistribución de posiciones:")
    print(f"Centro: {df['is_center'].sum()} ({df['is_center'].mean()*100:.1f}%)")
    print(f"Esquinas: {df['is_corner'].sum()} ({df['is_corner'].mean()*100:.1f}%)")
    print(f"Bordes: {df['is_edge'].sum()} ({df['is_edge'].mean()*100:.1f}%)")
    print(f"\nResultados guardados en: {output_file}") 