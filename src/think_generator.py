import random
from typing import Tuple

THINK_TEMPLATES = {
    "win": [
        "I detect a winning opportunity: I place at {move} and complete the line.",
        "This move at {move} secures a win for me.",
        "I can win by playing at {move}, completing my sequence.",
        "Playing at {move} allows me to finish a row, column, or diagonal.",
        "Victory is within reach. I move to {move} to win."
    ],
    "block": [
        "The opponent is threatening a win. I block at {move}.",
        "To prevent the opponent from winning, I must move to {move}.",
        "There's a danger of losing if I don't block at {move}.",
        "The opponent is one move away. I block them by playing at {move}.",
        "To avoid defeat, I make a defensive move at {move}."
    ],
    "neutral": [
        "There are no immediate threats or opportunities, so I choose {move} for strategic positioning.",
        "I play at {move} to control the center or setup future plays.",
        "No winning or blocking move is required, so I choose {move}.",
        "I take an open spot at {move} to improve my chances later.",
        "The board is balanced, I place at {move} for better positioning."
    ]
}

def generate_think_text(kind: str, move: Tuple[int, int]) -> str:
    """
    kind: "win", "block", or "neutral"
    move: Tuple (i, j) representing the move coordinates
    """
    template = random.choice(THINK_TEMPLATES[kind])
    move_str = f"{move[0]}-{move[1]}"
    return f"<think>{template.format(move=move_str)}</think>"
