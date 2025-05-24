import random
from typing import List, Tuple, Dict
from board_engine import (
    create_empty_board,
    is_valid_board,
    check_winner,
    is_draw,
    get_valid_moves,
    apply_move,
    next_player
)
from board_tokenizer import board_to_token_representation
from think_generator import generate_think_text

def generate_complete_game() -> List[Dict]:
    """
    Genera un juego completo de Tic Tac Toe y retorna una lista de estados
    con sus respectivos movimientos y razonamientos.
    """
    board = create_empty_board()
    game_states = []
    
    while True:
        current_player = next_player(board)
        valid_moves = get_valid_moves(board)
        
        if not valid_moves:
            break
            
        # Determinar el tipo de movimiento
        move_type = "neutral"
        for move in valid_moves:
            # Probar si es un movimiento ganador
            test_board = apply_move(board, current_player, move)
            if check_winner(test_board) == current_player:
                move_type = "win"
                break
                
            # Probar si es un movimiento de bloqueo
            opponent = 'O' if current_player == 'X' else 'X'
            test_board = apply_move(board, opponent, move)
            if check_winner(test_board) == opponent:
                move_type = "block"
                break
        
        # Seleccionar un movimiento válido
        move = random.choice(valid_moves)
        
        # Generar el estado actual
        state = {
            "board": board_to_token_representation(board),
            "think": generate_think_text(move_type, move),
            "move": f"<|move|><|{move[0]}-{move[1]}|><|end|>"
        }
        game_states.append(state)
        
        # Aplicar el movimiento
        board = apply_move(board, current_player, move)
        
        # Verificar si el juego ha terminado
        if check_winner(board) or is_draw(board):
            break
    
    return game_states

def generate_dataset(num_games: int = 5000) -> List[Dict]:
    """
    Genera un dataset completo con el número especificado de juegos.
    """
    dataset = []
    for _ in range(num_games):
        game_states = generate_complete_game()
        dataset.extend(game_states)
    return dataset

def save_dataset(dataset: List[Dict], filename: str = "tictac_dataset.json"):
    """
    Guarda el dataset en formato JSON.
    """
    import json
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    print("Generando dataset de 5,000 juegos...")
    dataset = generate_dataset(5000)
    print(f"Dataset generado con {len(dataset)} ejemplos")
    save_dataset(dataset)
    print("Dataset guardado en tictac_dataset.json") 