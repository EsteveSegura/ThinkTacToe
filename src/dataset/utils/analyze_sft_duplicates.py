import json
from collections import Counter
from typing import Dict, List
import re

def parse_board(board_str: str) -> str:
    """
    Convierte el string del tablero en un formato que pueda ser usado como clave en un diccionario
    """
    pattern = r'<\|(\d)-(\d)\|><\|([XOblank])\|>'
    matches = re.findall(pattern, board_str)
    
    # Creamos una matriz 3x3 con espacios vacíos
    board = [[' ' for _ in range(3)] for _ in range(3)]
    
    # Llenamos la matriz con los valores encontrados
    for row, col, value in matches:
        row, col = int(row), int(col)
        board[row][col] = value if value != 'blank' else ' '
    
    # Convertimos la matriz en un string para poder usarlo como clave
    return '\n'.join([''.join(row) for row in board])

def analyze_duplicates(dataset_path: str):
    """
    Analiza el dataset y encuentra tableros duplicados
    """
    # Cargamos el dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Contamos las ocurrencias de cada tablero
    board_counter = Counter()
    for state in dataset:
        board_str = parse_board(state['board'])
        board_counter[board_str] += 1
    
    # Encontramos los tableros que aparecen más de una vez
    duplicates = {board: count for board, count in board_counter.items() if count > 1}
    
    # Imprimimos los resultados
    print(f"\nTotal de tableros únicos: {len(board_counter)}")
    print(f"Total de tableros duplicados: {len(duplicates)}")
    print(f"Total de estados en el dataset: {len(dataset)}")
    
    if duplicates:
        print("\nTableros duplicados:")
        for board, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True):
            print(f"\nAparece {count} veces:")
            print(board)

    print(f"\nTotal de tableros únicos: {len(board_counter)}")
    print(f"Total de tableros duplicados: {len(duplicates)}")
    print(f"Total de estados en el dataset: {len(dataset)}")
if __name__ == "__main__":
    analyze_duplicates("tictactoe_dataset_dpo.json") 