import random
from typing import List, Tuple, Dict, Optional
from board_engine import (
    create_empty_board,
    check_winner,
    is_draw,
    get_valid_moves,
    apply_move,
    next_player
)
from board_tokenizer import board_to_token_representation
from think_generator import generate_think_text

def select_best_neutral(valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
    priorities = [
        [(1, 1)],
        [(0, 0), (0, 2), (2, 0), (2, 2)],
        [(0, 1), (1, 0), (1, 2), (2, 1)]
    ]
    for group in priorities:
        options = [m for m in group if m in valid_moves]
        if options:
            return random.choice(options)
    return random.choice(valid_moves)

def select_worst_neutral(valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
    priorities = [
        [(0, 1), (1, 0), (1, 2), (2, 1)],
        [(0, 0), (0, 2), (2, 0), (2, 2)],
        [(1, 1)]
    ]
    for group in priorities:
        options = [m for m in group if m in valid_moves]
        if options:
            return random.choice(options)
    return random.choice(valid_moves)

def generate_dpo_example() -> Optional[Dict]:
    board = create_empty_board()
    num_random_moves = random.randint(2, 6)
    for _ in range(num_random_moves):
        if check_winner(board) or is_draw(board):
            return None
        player = next_player(board)
        move = random.choice(get_valid_moves(board))
        board = apply_move(board, player, move)

    if check_winner(board) or is_draw(board):
        return None

    current_player = next_player(board)
    if current_player != 'X':
        return None

    valid_moves = get_valid_moves(board)
    optimal = None
    move_type = "neutral"

    for move in valid_moves:
        if check_winner(apply_move(board, 'X', move)) == 'X':
            move_type = "win"
            optimal = move
            break

    if not optimal:
        for move in valid_moves:
            if check_winner(apply_move(board, 'O', move)) == 'O':
                move_type = "block"
                optimal = move
                break

    if not optimal:
        optimal = select_best_neutral(valid_moves)
        move_type = "neutral"

    suboptimal_candidates = [m for m in valid_moves if m != optimal]
    if not suboptimal_candidates:
        return None

    bad_move = select_worst_neutral(suboptimal_candidates)

    return {
        "board": board_to_token_representation(board),
        "player": "X",
        "ok": [{
            "think": generate_think_text(move_type, optimal),
            "move": f"<|move|><|{optimal[0]}-{optimal[1]}|><|end|>"
        }],
        "ko": [{
            "think": generate_think_text("neutral", bad_move),
            "move": f"<|move|><|{bad_move[0]}-{bad_move[1]}|><|end|>"
        }]
    }

def generate_dpo_dataset(num_examples: int = 50000) -> List[Dict]:
    dataset = []
    while len(dataset) < num_examples:
        ex = generate_dpo_example()
        if ex:
            dataset.append(ex)
    return dataset

def save_dpo_dataset(dataset: List[Dict], filename: str = "tictac_dpo_dataset.json"):
    import json
    with open(filename, "w") as f:
        json.dump(dataset, f, indent=2)

def save_dpo_dataset_text(dataset: List[Dict], filename: str = "tictac_dpo_dataset.jsonl"):
    import json
    with open(filename, "w") as f:
        for ex in dataset:
            text = (
                "<|board_start|>\n" + ex["board"] + "\n<|board_end|>\n"
                "<|player|>X\n"
                + ex["ok"][0]["think"] + "\n" + ex["ok"][0]["move"] + "\n"
                + "<|vs|>\n"
                + ex["ko"][0]["think"] + "\n" + ex["ko"][0]["move"]
            )
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    dpo_dataset = generate_dpo_dataset(10000)
    save_dpo_dataset(dpo_dataset)
    save_dpo_dataset_text(dpo_dataset)
    print("DPO dataset saved.")
