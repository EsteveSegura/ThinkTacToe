from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import re

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

# Configuración de modelos a probar
MODELS_CONFIG = [
    # Modelos 0.5B
        {
        'name': 'qwen2.5-0.5b-tictactoe-sft-nothink-minmax',
        'path': '/home/ThinkTacToe/qwen2.5-0.5b-tictactoe-sft-nothink-minmax/checkpoint-852',
        'has_think': False
    },

]

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
    Soporta formatos:
      - <|move|><|2-2|><|end|>
      - <|2-2|><|end|>
      - 2-2|><|end|>
      - 2-2<|end|>
    """
    try:
        # Buscar patrón de coordenadas d-d en el texto
        match = re.search(r"(\d)-(\d)", model_output)
        if match:
            row, col = int(match.group(1)), int(match.group(2))
            return (row, col)
        return None
    except:
        return None

def get_optimal_moves(board: list) -> list:
    """
    Determina cuáles son los movimientos óptimos para el tablero actual.
    Retorna una lista de tuplas con las coordenadas de los movimientos óptimos.
    """
    valid_moves = get_valid_moves(board)
    if not valid_moves:
        return []
    
    optimal_moves = []
    
    for move in valid_moves:
        # Aplicar el movimiento
        new_board = apply_move(board, 'X', move)
        
        # Verificar si es un movimiento ganador
        if check_winner(new_board) == 'X':
            optimal_moves.append(move)
            continue
        
        # Verificar si bloquea un movimiento ganador del oponente
        opponent_board = apply_move(board, 'O', move)
        if check_winner(opponent_board) == 'O':
            optimal_moves.append(move)
            continue
    
    # Si no hay movimientos ganadores ni bloqueadores, considerar el centro como óptimo
    if not optimal_moves and (1, 1) in valid_moves:
        optimal_moves.append((1, 1))
    
    # Si no hay movimientos claramente óptimos, todos los movimientos válidos son considerados óptimos (neutros)
    if not optimal_moves:
        optimal_moves = valid_moves
    
    return optimal_moves

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
            'is_edge': False,
            'optimal_move': False
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
            'is_edge': False,
            'optimal_move': False
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
    
    # Verificar si es un movimiento óptimo
    optimal_moves = get_optimal_moves(board)
    optimal_move = move in optimal_moves
    
    return {
        'is_valid': True,
        'is_winning': is_winning,
        'is_blocking': is_blocking,
        'is_center': is_center,
        'is_corner': is_corner,
        'is_edge': is_edge,
        'optimal_move': optimal_move
    }

def infer(model, tokenizer, prompt: str, max_new_tokens: int = 600):
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

def create_prompt(board: list, has_think: bool) -> str:
    """
    Crea el prompt para el modelo según si tiene capacidad de pensar o no.
    """
    board_repr = board_to_token_representation(board)
    
    # Calcular el turno basado en el número de movimientos realizados
    moves_count = sum(1 for row in board for cell in row if cell is not None)
    turn = moves_count + 1
    
    # Determinar el símbolo del jugador (X siempre va primero)
    player = 'X'
    
    if has_think:
        # Modelos con capacidad de pensar
        return f"<|board_start|>\n{board_repr}\n<|board_end|>\n<|turn|>{turn}\n<|symbol|>{player}\n<|player_think>"
    else:
        # Modelos sin capacidad de pensar (nothink)
        return f"<|board_start|>\n{board_repr}\n<|board_end|>\n<|turn|>{turn}\n<|symbol|>{player}\n<|move|><|"

def test_model(model_config: dict, num_games: int = 10) -> pd.DataFrame:
    """
    Prueba un modelo específico y retorna los resultados.
    """
    model_name = model_config['name']
    model_path = model_config['path']
    has_think = model_config['has_think']
    
    print(f"\n{'='*60}")
    print(f"Probando modelo: {model_name}")
    print(f"Ruta: {model_path}")
    print(f"Tiene capacidad de pensar: {has_think}")
    print(f"{'='*60}")
    
    # Cargar modelo y tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
        print(f"Modelo cargado exitosamente")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return pd.DataFrame()
    
    results = []
    
    for game in range(num_games):
        if (game + 1) % 20 == 0:
            print(f"  Progreso: {game + 1}/{num_games}")
        
        # Crear tablero aleatorio
        board = create_random_board()
        
        # Crear prompt según el tipo de modelo
        prompt = create_prompt(board, has_think)
        
        # Obtener respuesta del modelo
        model_output = infer(model, tokenizer, prompt)
        
        # Mostrar prompt y respuesta en consola
        print(f"\n--- Juego {game + 1} ---")
        print(f"Prompt:")
        print(prompt)
        print(f"\nRespuesta del modelo:")
        print(model_output)
        print(f"--- Fin Juego {game + 1} ---\n")
        
        # Extraer el movimiento
        move = parse_model_move(model_output)
        
        # Evaluar el movimiento
        evaluation = evaluate_move(board, move)
        
        # Guardar resultados
        result = {
            'model_name': model_name,
            'model_path': model_path,
            'has_think': has_think,
            'game_id': game + 1,
            'initial_board': board_to_string(board),
            'prompt': prompt,
            'model_output': model_output,
            'move': str(move) if move else 'None',
            **evaluation
        }
        results.append(result)
    
    # Crear DataFrame
    df = pd.DataFrame(results)
    
    # Mostrar resumen del modelo
    print(f"\nResumen para {model_name}:")
    print(f"  Total de juegos: {len(df)}")
    print(f"  Movimientos válidos: {df['is_valid'].sum()} ({df['is_valid'].mean()*100:.1f}%)")
    print(f"  Movimientos ganadores: {df['is_winning'].sum()} ({df['is_winning'].mean()*100:.1f}%)")
    print(f"  Movimientos bloqueadores: {df['is_blocking'].sum()} ({df['is_blocking'].mean()*100:.1f}%)")
    print(f"  Movimientos óptimos: {df['optimal_move'].sum()} ({df['optimal_move'].mean()*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Crear directorio de resultados si no existe
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Generar timestamp para los archivos
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Probar todos los modelos
    all_results = []
    
    for model_config in MODELS_CONFIG:
        try:
            df = test_model(model_config, num_games=500)
            if not df.empty:
                all_results.append(df)
                
                # Guardar resultados individuales del modelo
                model_name_clean = model_config['name'].replace('/', '_').replace('-', '_')
                individual_file = results_dir / f'{model_name_clean}_evaluation_{timestamp}.csv'
                df.to_csv(individual_file, index=False)
                print(f"Resultados guardados en: {individual_file}")
                
        except Exception as e:
            print(f"Error al probar modelo {model_config['name']}: {e}")
            continue
    
    # Combinar todos los resultados
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Guardar resultados combinados
        combined_file = results_dir / f'all_models_evaluation_{timestamp}.csv'
        combined_df.to_csv(combined_file, index=False)
        
        # Mostrar resumen general
        print(f"\n{'='*80}")
        print("RESUMEN GENERAL DE TODOS LOS MODELOS")
        print(f"{'='*80}")
        
        for model_name in combined_df['model_name'].unique():
            model_data = combined_df[combined_df['model_name'] == model_name]
            print(f"\n{model_name}:")
            print(f"  Juegos: {len(model_data)}")
            print(f"  Válidos: {model_data['is_valid'].mean()*100:.1f}%")
            print(f"  Ganadores: {model_data['is_winning'].mean()*100:.1f}%")
            print(f"  Bloqueadores: {model_data['is_blocking'].mean()*100:.1f}%")
            print(f"  Óptimos: {model_data['optimal_move'].mean()*100:.1f}%")
        
        print(f"\nResultados combinados guardados en: {combined_file}")
    else:
        print("No se pudieron procesar resultados de ningún modelo.") 