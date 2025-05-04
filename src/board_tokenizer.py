from typing import Optional, List

Board = List[List[Optional[str]]]

def board_to_token_representation(board: Board) -> str:
    lines = []
    for i, row in enumerate(board):
        line = []
        for j, cell in enumerate(row):
            value = "blank" if cell is None else cell
            line.append(f"<|{i}-{j}|><|{value}|>")
        lines.append(" ".join(line))
    return "\n".join(lines)
