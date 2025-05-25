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

def generate_complete_game() -> List[dict]:
    """
    Genera un juego completo de tres en raya.
    """
    board = create_empty_board()
    states = []
    current_player = 'X'
    
    # Verificar que el tablero inicial es válido
    assert is_valid_board(board), "Tablero inicial inválido"
    
    while True:
        # Verificar si el juego ya terminó
        if check_winner(board) or is_draw(board):
            break

        valid_moves = get_valid_moves(board)
        if not valid_moves:
            break

        move_type = "neutral"
        selected_move = None

        # Ver si puedo ganar
        for move in valid_moves:
            test_board = apply_move(board, current_player, move)
            if check_winner(test_board) == current_player:
                move_type = "win"
                selected_move = move
                break

        # Ver si debo bloquear
        if selected_move is None:
            opponent = 'O' if current_player == 'X' else 'X'
            for move in valid_moves:
                test_board = apply_move(board, opponent, move)
                if check_winner(test_board) == opponent:
                    move_type = "block"
                    selected_move = move
                    break

        # Si no hay win ni block, elijo al azar
        if selected_move is None:
            selected_move = random.choice(valid_moves)

        i, j = selected_move
        # ✅ Verificar explícitamente que la celda está libre
        if board[i][j] is not None:
            continue  # debería ser imposible, pero por seguridad

        # Crear el estado antes de aplicar el movimiento
        state = {
            "board": board_to_token_representation(board),
            "think": generate_think_text(move_type, selected_move),
            "move": f"<|move|><|{i}-{j}|><|end|>"
        }

        # Aplicar el movimiento
        board = apply_move(board, current_player, selected_move)

        # ✅ Verificar si el juego terminó DESPUÉS del movimiento
        if check_winner(board) or is_draw(board):
            break  # no se guarda este movimiento, porque el juego terminó

        states.append(state)

        # Cambiar de jugador
        current_player = 'O' if current_player == 'X' else 'X'

    return states

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
