import json
from collections import Counter
from typing import Dict, List
import re

def parse_board(board_str: str) -> str:
    pattern = r'<\|(\d)-(\d)\|><\|([XOblank])\|>'
    matches = re.findall(pattern, board_str)
    
    board = [[' ' for _ in range(3)] for _ in range(3)]
    
    for row, col, value in matches:
        row, col = int(row), int(col)
        board[row][col] = value if value != 'blank' else ' '
    
    return '\n'.join([''.join(row) for row in board])

def analyze_duplicates(dataset_path: str):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    board_counter = Counter()
    for state in dataset:
        board_str = parse_board(state['board'])
        board_counter[board_str] += 1
    
    duplicates = {board: count for board, count in board_counter.items() if count > 1}
    
    print(f"\nTotal de tableros únicos: {len(board_counter)}")
    print(f"Total de tableros duplicados: {len(duplicates)}")
    print(f"Total de estados en el dataset: {len(dataset)}")
    
    if duplicates:
        print("\nTableros duplicados:")
        for board, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True):
            print(f"\nAparece {count} veces:")
            print(board)
            matching_states = [state for state in dataset if parse_board(state['board']) == board]
            print("\nEjemplos de movimientos para este tablero:")
            for state in matching_states[:3]:
                print(f"OK: {state['ok'][0]['move'] if state['ok'] else 'N/A'}")
                print(f"KO: {state['ko'][0]['move'] if state['ko'] else 'N/A'}")
                print("---")
    
    # Resumen final
    total_duplicates = sum(count - 1 for count in duplicates.values())
    print(f"\nRESUMEN FINAL:")
    print(f"Total de tableros duplicados (solo board): {total_duplicates}")

if __name__ == "__main__":
    analyze_duplicates("tictactoe_dataset_dpo.json") 