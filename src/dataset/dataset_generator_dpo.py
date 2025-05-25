import random
import json
from typing import List, Dict, Tuple, Optional
from board_engine import (
    create_empty_board,
    is_valid_board,
    check_winner,
    is_draw,
    get_valid_moves,
    apply_move,
    next_player,
    Player,
    Board
)
from board_tokenizer import board_to_token_representation
from think_generator import generate_think_text


def categorize_moves(board: Board, current_player: Player) -> Dict[str, List[Tuple[int,int]]]:
    """
    Clasifica los movimientos válidos en ganadores, bloqueadores o neutrales.
    """
    valid_moves = get_valid_moves(board)
    win_moves: List[Tuple[int,int]] = []
    block_moves: List[Tuple[int,int]] = []
    neutral_moves: List[Tuple[int,int]] = []
    opponent = 'O' if current_player == 'X' else 'X'

    for move in valid_moves:
        # Verificar movimiento ganador
        test_board = apply_move(board, current_player, move)
        if check_winner(test_board) == current_player:
            win_moves.append(move)
        else:
            # Verificar movimiento de bloqueo
            test_board_op = apply_move(board, opponent, move)
            if check_winner(test_board_op) == opponent:
                block_moves.append(move)
            else:
                neutral_moves.append(move)

    return {"win": win_moves, "block": block_moves, "neutral": neutral_moves}


def get_optimal_moves(categories: Dict[str, List[Tuple[int,int]]]) -> List[Tuple[int,int]]:
    """
    Devuelve la lista de movimientos óptimos según prioridad:
    win > block > center > corners > rest neutrales.
    """
    if categories["win"]:
        return categories["win"]
    if categories["block"]:
        return categories["block"]

    neutral = categories["neutral"]
    optimal: List[Tuple[int,int]] = []
    center = (1, 1)

    # Priorizar el centro
    if center in neutral:
        optimal.append(center)

    # Luego las esquinas
    corners = [(0,0), (0,2), (2,0), (2,2)]
    for c in corners:
        if c in neutral:
            optimal.append(c)

    # Si no hay centro ni esquina, usar todos neutrales
    return optimal or neutral


def generate_dpo_example(board: Board, current_player: Player) -> Optional[Dict]:
    """
    Genera un ejemplo DPO para un tablero dado:
    - ok: lista de respuestas correctas
    - ko: lista de respuestas incorrectas
    """
    # No generar ejemplos en tableros terminales
    if check_winner(board) or is_draw(board):
        return None

    categories = categorize_moves(board, current_player)
    optimal_moves = get_optimal_moves(categories)
    if not optimal_moves:
        return None

    # Generar casos OK
    ok: List[Dict[str, str]] = []
    for move in optimal_moves:
        kind = (
            "win" if move in categories["win"] else
            "block" if move in categories["block"] else
            "neutral"
        )
        think = generate_think_text(kind, move)
        move_token = f"<|move|><|{move[0]}-{move[1]}|><|end|>"
        ok.append({"think": think, "move": move_token})

    # Generar casos KO (no óptimos)
    all_valid = get_valid_moves(board)
    non_opt = [m for m in all_valid if m not in optimal_moves]
    random.shuffle(non_opt)

    # Tomar hasta doble de negativos que positivos
    ko: List[Dict[str, str]] = []
    for move in non_opt[: len(ok) * 2]:
        kind = (
            "win" if move in categories["win"] else
            "block" if move in categories["block"] else
            "neutral"
        )
        think = generate_think_text(kind, move)
        move_token = f"<|move|><|{move[0]}-{move[1]}|><|end|>"
        ko.append({"think": think, "move": move_token})

    return {
        "board": board_to_token_representation(board),
        "ok": ok,
        "ko": ko
    }


def generate_dpo_dataset(num_examples: int = 50000) -> List[Dict]:
    """
    Genera un dataset DPO con tableros en estados aleatorios no terminales.
    """
    dataset: List[Dict] = []

    while len(dataset) < num_examples:
        # Generar estado de tablero inicial vacío
        board = create_empty_board()

        # Aplicar entre 0 y 4 jugadas aleatorias
        num_moves = random.randint(0, 4)
        for _ in range(num_moves):
            if check_winner(board) or is_draw(board):
                break
            player = next_player(board)
            moves = get_valid_moves(board)
            board = apply_move(board, player, random.choice(moves))

        # Generar ejemplo DPO si válido
        player = next_player(board)
        example = generate_dpo_example(board, player)
        if example and example["ok"] and example["ko"]:
            dataset.append(example)

    return dataset


def save_dataset(dataset: List[Dict], filename: str = "tictac_dpo_dataset.json"):
    """
    Guarda el dataset en formato JSON.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("Generando dataset DPO de TicTacToe...")
    ds = generate_dpo_dataset(50000)
    print(f"Guardando {len(ds)} ejemplos en tictac_dpo_dataset.json")
    save_dataset(ds)
    print("¡Listo!")
