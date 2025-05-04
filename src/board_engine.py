# Creamos un motor básico de Tic Tac Toe (3x3) en Python
from typing import List, Optional, Tuple
import random

Player = str  # 'X' (modelo) o 'O' (humano)
Board = List[List[Optional[Player]]]  # 3x3 con 'X', 'O' o None

def create_empty_board() -> Board:
    return [[None for _ in range(3)] for _ in range(3)]

def print_board(board: Board) -> None:
    for row in board:
        print([" " if cell is None else cell for cell in row])
    print()

def is_valid_board(board: Board) -> bool:
    x_count = sum(cell == 'X' for row in board for cell in row)
    o_count = sum(cell == 'O' for row in board for cell in row)
    return x_count == o_count or x_count == o_count + 1

def check_winner(board: Board) -> Optional[Player]:
    lines = []

    # Filas y columnas
    for i in range(3):
        lines.append(board[i])  # fila
        lines.append([board[0][i], board[1][i], board[2][i]])  # columna

    # Diagonales
    lines.append([board[0][0], board[1][1], board[2][2]])
    lines.append([board[0][2], board[1][1], board[2][0]])

    for line in lines:
        if line[0] is not None and line.count(line[0]) == 3:
            return line[0]
    return None

def is_draw(board: Board) -> bool:
    return all(cell is not None for row in board for cell in row) and check_winner(board) is None

def get_valid_moves(board: Board) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] is None]

def apply_move(board: Board, player: Player, move: Tuple[int, int]) -> Board:
    new_board = [row[:] for row in board]
    i, j = move
    new_board[i][j] = player
    return new_board

def next_player(board: Board) -> Player:
    x_count = sum(cell == 'X' for row in board for cell in row)
    o_count = sum(cell == 'O' for row in board for cell in row)
    return 'X' if x_count == o_count else 'O'

# Funciones para generar situaciones específicas
def generate_win_scenario() -> Board:
    board = create_empty_board()
    # X está a punto de ganar
    board[0][0], board[0][1] = 'X', 'X'
    board[1][0], board[1][1] = 'O', 'O'
    return board

def generate_block_scenario() -> Board:
    board = create_empty_board()
    # O está a punto de ganar, X debe bloquear
    board[0][0], board[0][1] = 'O', 'O'
    board[1][0], board[1][1] = 'X', 'X'
    return board

def generate_neutral_scenario() -> Board:
    board = create_empty_board()
    board[0][0] = 'X'
    board[1][1] = 'O'
    return board

# Probamos el motor
win_board = generate_win_scenario()
block_board = generate_block_scenario()
neutral_board = generate_neutral_scenario()


