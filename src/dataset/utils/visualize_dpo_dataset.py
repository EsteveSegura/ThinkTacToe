import json
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import re
import os
from datetime import datetime

def parse_board(board_str: str) -> List[List[str]]:
    board = [[None for _ in range(3)] for _ in range(3)]
    
    pattern = r'<\|(\d)-(\d)\|><\|([XOblank])\|>'
    matches = re.findall(pattern, board_str)
    
    for row, col, value in matches:
        row, col = int(row), int(col)
        board[row][col] = value if value != 'blank' else None
    
    return board

def parse_move(move_str: str) -> tuple:
    pattern = r'<\|(\d)-(\d)\|>'
    match = re.search(pattern, move_str)
    if match:
        row, col = map(int, match.groups())
        return row, col
    return None

def visualize_board(board: List[List[str]], title: str = "", ok_move: str = None, ko_move: str = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Dibujamos el tablero
    for i in range(4):
        ax.plot([0, 3], [i, i], 'k-', linewidth=2)
        ax.plot([i, i], [0, 3], 'k-', linewidth=2)
    
    # Dibujamos las X y O existentes
    for i in range(3):
        for j in range(3):
            if board[i][j] == 'X':
                ax.plot([j + 0.2, j + 0.8], [2.8 - i, 2.2 - i], 'y-', linewidth=3)
                ax.plot([j + 0.2, j + 0.8], [2.2 - i, 2.8 - i], 'y-', linewidth=3)
            elif board[i][j] == 'O':
                circle = plt.Circle((j + 0.5, 2.5 - i), 0.3, fill=False, color='blue', linewidth=3)
                ax.add_artist(circle)
    
    # Dibujamos el movimiento correcto (ok) en verde
    if ok_move:
        ok_pos = parse_move(ok_move)
        if ok_pos:
            row, col = ok_pos
            ax.plot([col + 0.2, col + 0.8], [2.8 - row, 2.2 - row], 'g-', linewidth=3)
            ax.plot([col + 0.2, col + 0.8], [2.2 - row, 2.8 - row], 'g-', linewidth=3)
    
    # Dibujamos el movimiento incorrecto (ko) en rojo
    if ko_move:
        ko_pos = parse_move(ko_move)
        if ko_pos:
            row, col = ko_pos
            ax.plot([col + 0.2, col + 0.8], [2.8 - row, 2.2 - row], 'r-', linewidth=3)
            ax.plot([col + 0.2, col + 0.8], [2.2 - row, 2.8 - row], 'r-', linewidth=3)
    
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylim(-0.1, 3.1)
    ax.axis('off')
    if title:
        ax.set_title(title, pad=20)
    
    return fig

def generate_visualization(dataset: List[Dict], num_samples: int, save_path: str):
    random_states = random.sample(dataset, min(num_samples, len(dataset)))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(6*num_samples, 6))
    if num_samples == 1:
        axes = [axes]
    
    for i, state in enumerate(random_states):
        board = parse_board(state['board'])
        ok_move = state['ok'][0]['move'] if state['ok'] else None
        ko_move = state['ko'][0]['move'] if state['ko'] else None
        
        title = f"Estado {i+1}\nOK: {state['ok'][0]['think'] if state['ok'] else 'N/A'}\nKO: {state['ko'][0]['think'] if state['ko'] else 'N/A'}"
        visualize_board(board, title, ok_move, ko_move)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300, format='jpg')
    plt.close()
    print(f"Visualización guardada en {save_path}")

def visualize_random_states(dataset_path: str, num_samples: int = 5, num_visualizations: int = 3, save_path: str = "tmp/dpo_visualization"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    for i in range(num_visualizations):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_save_path = f"{save_path}_{i+1}_{timestamp}.jpg"
        generate_visualization(dataset, num_samples, full_save_path)

if __name__ == "__main__":
    visualize_random_states("tictac_dpo_dataset.json", num_samples=5, num_visualizations=12) 