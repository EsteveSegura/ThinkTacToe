import re

def parse_board(board_str):
    board = [['' for _ in range(3)] for _ in range(3)]
    pattern = r"<\|(\d)-(\d)\|><\|(.*?)\|>"
    for row, col, val in re.findall(pattern, board_str):
        board[int(row)][int(col)] = val
    return board

def parse_move(move_str):
    match = re.search(r"<\|move\|><\|(\d)-(\d)\|>", move_str)
    return (int(match.group(1)), int(match.group(2))) if match else None

def render_board(board_str, move_str=None):
    board = parse_board(board_str)
    move = parse_move(move_str) if move_str else None
    print("\n+---+---+---+")
    for i, row in enumerate(board):
        row_str = ""
        for j, cell in enumerate(row):
            symbol = cell if cell != "blank" else " "
            if move == (i, j):
                row_str += f"|{symbol}*"
            else:
                row_str += f"| {symbol} "
        print(row_str + "|")
        print("+---+---+---+")

data = {
    "board": "<|0-0|><|O|> <|0-1|><|blank|> <|0-2|><|X|>\n<|1-0|><|blank|> <|1-1|><|X|> <|1-2|><|blank|>\n<|2-0|><|O|> <|2-1|><|blank|> <|2-2|><|blank|>",
    "move": "<|move|><|1-0|><|end|>"
}

render_board(data["board"], data["move"])
