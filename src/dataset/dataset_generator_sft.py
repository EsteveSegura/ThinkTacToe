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
from think_generator_llm import generate_think_llm

# Flag to control whether to use LLM for thought generation
LLM_THINK = False
# Flag to skip think generation completely
NO_THINK = False

def select_best_neutral(valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
    # Priorizamos centro, luego esquinas, luego laterales
    priorities = [
        [(1, 1)],  # centro
        [(0, 0), (0, 2), (2, 0), (2, 2)],  # esquinas
        [(0, 1), (1, 0), (1, 2), (2, 1)]   # laterales
    ]

    for group in priorities:
        options = [m for m in group if m in valid_moves]
        if options:
            return random.choice(options)
    
    return random.choice(valid_moves)

def generate_complete_game() -> List[dict]:
    """
    Genera un estado de juego aleatorio válido, y guarda un único paso del jugador 'X'.
    El tablero usa coordenadas donde:
    - row: 0=top, 1=middle, 2=bottom
    - col: 0=left, 1=center, 2=right
    """
    board = create_empty_board()

    # ✅ Aplica entre 2 y 6 jugadas aleatorias
    num_random_moves = random.randint(2, 6)
    for _ in range(num_random_moves):
        if check_winner(board) or is_draw(board):
            break
        player = next_player(board)
        move = random.choice(get_valid_moves(board))
        board = apply_move(board, player, move)

    # Verifica si el tablero es terminal
    if check_winner(board) or is_draw(board):
        return []

    # Ahora le toca a un jugador (X u O)
    current_player = next_player(board)

    if current_player != 'X':
        return []  # Solo nos interesan jugadas de X

    valid_moves = get_valid_moves(board)
    move_type = "neutral"
    selected_move = None

    # Ver si puede ganar
    for move in valid_moves:
        test_board = apply_move(board, 'X', move)
        if check_winner(test_board) == 'X':
            move_type = "win"
            selected_move = move
            break

    # Ver si debe bloquear
    if selected_move is None:
        for move in valid_moves:
            test_board = apply_move(board, 'O', move)
            if check_winner(test_board) == 'O':
                move_type = "block"
                selected_move = move
                break

    if selected_move is None:
        selected_move = select_best_neutral(valid_moves)

    i, j = selected_move
    if board[i][j] is not None:
        return []

    # Generate thought using either LLM or original method
    if NO_THINK:
        think = ""
    elif LLM_THINK:
        think = generate_think_llm(board, selected_move)
    else:
        think = generate_think_text(move_type, selected_move)

    state = {
        "board": board_to_token_representation(board),
        "player": "X",
        "think": think,
        "move": f"<|move|><|{i}-{j}|><|end|>"
    }

    return [state]

def generate_dataset(num_games: int = 5000) -> List[Dict]:
    print(f"Generating: {num_games} games...")
    dataset = []
    attempts = 0
    while len(dataset) < num_games:
        print(f"Generating game {len(dataset) + 1} of {num_games}")
        states = generate_complete_game()
        if states:
            dataset.extend(states)
        attempts += 1
        if attempts > num_games * 5:
            print("Stopping early to avoid infinite loop.")
            break
    return dataset

def save_dataset(dataset: List[Dict], filename: str = None):
    if filename is None:
        # Build filename based on configuration
        config = []
        if NO_THINK:
            config.append("nothink")
        elif LLM_THINK:
            config.append("llm")
        else:
            config.append("template")
        config_str = "_".join(config)
        filename = f"tictactoe_sft_{config_str}.json"

    import json
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ SFT dataset saved as {filename}")

def save_dataset_text(dataset: List[Dict], filename: str = None):
    """
    Guarda el dataset como JSONL en formato {"text": "..."} por línea,
    incluyendo delimitadores <|board_start|> y <|board_end|>.
    """
    if filename is None:
        # Build filename based on configuration
        config = []
        if NO_THINK:
            config.append("nothink")
        elif LLM_THINK:
            config.append("llm")
        else:
            config.append("template")
        config_str = "_".join(config)
        filename = f"tictactoe_sft_{config_str}.jsonl"

    import json
    with open(filename, "w", encoding="utf-8") as f:
        for example in dataset:
            board = example["board"]
            player = example.get("player", "X")
            think = example["think"]
            move = example["move"]

            text = (
                "<|board_start|>\n"
                f"{board}\n"
                "<|board_end|>\n"
                f"<|player|>{player}\n"
            )
            
            if not NO_THINK:
                text += f"{think}\n"
            
            text += f"{move}"

            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"✅ SFT dataset saved as {filename}")

if __name__ == "__main__":
    dataset = generate_dataset(1)
    print(f"Dataset generated with {len(dataset)} examples")
    save_dataset(dataset)
    save_dataset_text(dataset)
