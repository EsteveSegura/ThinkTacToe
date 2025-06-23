#!/usr/bin/env python3
"""
Script para probar la nueva función de evaluación de movimientos
"""

import sys
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.dataset.board_engine import (
    create_empty_board,
    apply_move,
    get_valid_moves,
    check_winner
)

def extract_board_from_prompt(prompt):
    """Extrae el tablero del prompt para evaluar el movimiento"""
    try:
        # Buscar la sección del tablero entre <|board_start|> y <|board_end|>
        board_start = prompt.find("<|board_start|>")
        board_end = prompt.find("<|board_end|>")
        
        if board_start == -1 or board_end == -1:
            return None
            
        board_text = prompt[board_start:board_end]
        
        # Crear tablero vacío
        board = create_empty_board()
        
        # Parsear las líneas del tablero
        lines = board_text.split('\n')[1:-1]  # Excluir las etiquetas de inicio/fin
        
        for i, line in enumerate(lines):
            if i >= 3:  # Solo las primeras 3 líneas
                break
                
            # Buscar patrones como <|0-0|><|X|> <|0-1|><|blank|> <|0-2|><|O|>
            import re
            cell_pattern = r'<\|(\d)-(\d)\|><\|([^|]+)\|>'
            matches = re.findall(cell_pattern, line)
            
            for match in matches:
                row, col, value = int(match[0]), int(match[1]), match[2]
                if value == 'X':
                    board[row][col] = 'X'
                elif value == 'O':
                    board[row][col] = 'O'
                # 'blank' se mantiene como None
        
        return board
    except Exception as e:
        print(f"Error parsing board: {e}")
        return None

def evaluate_move_quality(board, move):
    """Evalúa la calidad del movimiento en el contexto del tablero de manera más inteligente"""
    if not board or not move:
        return 0.0
    
    row, col = move
    
    # Verificar si el movimiento es válido
    valid_moves = get_valid_moves(board)
    if move not in valid_moves:
        return -1.0  # Movimiento inválido
    
    # Aplicar el movimiento
    new_board = apply_move(board, 'X', move)
    
    # Verificar si es un movimiento ganador
    if check_winner(new_board) == 'X':
        return 1.0  # Movimiento ganador
    
    # Verificar si bloquea un movimiento ganador del oponente
    opponent_board = apply_move(board, 'O', move)
    if check_winner(opponent_board) == 'O':
        return 0.8  # Movimiento bloqueador
    
    # Evaluación estratégica más sofisticada
    score = 0.0
    threat_created = False  # Inicializar la variable
    
    # Verificar si el centro está disponible y es una buena opción
    center_available = (1, 1) in valid_moves
    is_center_move = move == (1, 1)
    
    # Si el centro está disponible y no lo tomamos, penalizar ligeramente
    if center_available and not is_center_move:
        # Solo penalizar si no hay una razón estratégica clara
        # Verificar si nuestro movimiento crea una amenaza
        threat_created = False
        
        # Verificar si creamos una línea de dos en fila
        for direction in [(0, 1), (1, 0), (1, 1), (1, -1)]:  # horizontal, vertical, diagonales
            count = 1  # Contar nuestro movimiento
            # Contar en dirección positiva
            for step in range(1, 3):
                new_row, new_col = row + step * direction[0], col + step * direction[1]
                if 0 <= new_row < 3 and 0 <= new_col < 3 and new_board[new_row][new_col] == 'X':
                    count += 1
                else:
                    break
            # Contar en dirección negativa
            for step in range(1, 3):
                new_row, new_col = row - step * direction[0], col - step * direction[1]
                if 0 <= new_row < 3 and 0 <= new_col < 3 and new_board[new_row][new_col] == 'X':
                    count += 1
                else:
                    break
            
            if count >= 2:
                threat_created = True
                break
        
        if not threat_created:
            score -= 0.1  # Penalización menor por no tomar el centro sin razón
    
    # Recompensar movimientos que crean amenazas
    if threat_created:
        score += 0.3
    
    # Recompensar movimientos en esquinas (estratégicamente buenos)
    if move in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        score += 0.2
    
    # Recompensar movimientos en el centro (pero no excesivamente)
    if is_center_move:
        score += 0.1  # Recompensa menor que antes
    
    # Recompensar movimientos que bloquean amenazas del oponente
    # Verificar si el oponente tiene dos en línea en cualquier dirección
    opponent_threats_blocked = 0
    for direction in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        count = 0
        for step in range(-2, 3):
            new_row, new_col = row + step * direction[0], col + step * direction[1]
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                if board[new_row][new_col] == 'O':
                    count += 1
                elif board[new_row][new_col] == 'X':
                    count = 0  # Reset si encontramos nuestra pieza
                    break
        if count >= 2:
            opponent_threats_blocked += 1
    
    if opponent_threats_blocked > 0:
        score += 0.2 * opponent_threats_blocked
    
    # Recompensa base para movimientos válidos
    base_reward = 0.3
    
    return base_reward + score

def print_board(board):
    """Imprime el tablero de forma legible"""
    for i, row in enumerate(board):
        row_str = []
        for j, cell in enumerate(row):
            if cell is None:
                row_str.append(f"({i},{j})")
            else:
                row_str.append(f" {cell} ")
        print(" ".join(row_str))
    print()

def test_move_quality():
    """Prueba la función de evaluación de movimientos con diferentes escenarios"""
    
    test_cases = [
        {
            "name": "Tablero vacío - todas las posiciones",
            "prompt": "<|board_start|>\n<|0-0|><|blank|> <|0-1|><|blank|> <|0-2|><|blank|>\n<|1-0|><|blank|> <|1-1|><|blank|> <|1-2|><|blank|>\n<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>\n<|board_end|>\n<|player|>X",
            "moves": [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
        },
        {
            "name": "Centro ocupado por O",
            "prompt": "<|board_start|>\n<|0-0|><|blank|> <|0-1|><|blank|> <|0-2|><|blank|>\n<|1-0|><|blank|> <|1-1|><|O|> <|1-2|><|blank|>\n<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>\n<|board_end|>\n<|player|>X",
            "moves": [(0,0), (0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]
        },
        {
            "name": "Movimiento ganador disponible",
            "prompt": "<|board_start|>\n<|0-0|><|X|> <|0-1|><|X|> <|0-2|><|blank|>\n<|1-0|><|blank|> <|1-1|><|O|> <|1-2|><|blank|>\n<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>\n<|board_end|>\n<|player|>X",
            "moves": [(0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]
        },
        {
            "name": "Bloqueo de amenaza del oponente",
            "prompt": "<|board_start|>\n<|0-0|><|O|> <|0-1|><|O|> <|0-2|><|blank|>\n<|1-0|><|blank|> <|1-1|><|X|> <|1-2|><|blank|>\n<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>\n<|board_end|>\n<|player|>X",
            "moves": [(0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]
        }
    ]
    
    print("=== PRUEBA DE EVALUACIÓN DE MOVIMIENTOS ===\n")
    
    for test_case in test_cases:
        print(f"📋 {test_case['name']}")
        print("-" * 50)
        
        board = extract_board_from_prompt(test_case['prompt'])
        if board:
            print("Tablero actual:")
            print_board(board)
            
            print("Evaluación de movimientos:")
            for move in test_case['moves']:
                quality = evaluate_move_quality(board, move)
                move_type = "Centro" if move == (1,1) else "Esquina" if move in [(0,0), (0,2), (2,0), (2,2)] else "Lateral"
                print(f"  {move}: {quality:.3f} ({move_type})")
            
            # Encontrar el mejor movimiento
            best_move = max(test_case['moves'], key=lambda m: evaluate_move_quality(board, m))
            best_quality = evaluate_move_quality(board, best_move)
            print(f"\n🎯 Mejor movimiento: {best_move} (calidad: {best_quality:.3f})")
            
            # Verificar si hay sesgo hacia el centro
            center_quality = None
            for move in test_case['moves']:
                if move == (1, 1):
                    center_quality = evaluate_move_quality(board, move)
                    break
            
            if center_quality is not None:
                if center_quality == best_quality:
                    print("⚠️  El centro está empatado como mejor opción")
                elif center_quality > best_quality * 0.9:  # Si está muy cerca
                    print("⚠️  El centro tiene una calidad muy alta")
                else:
                    print("✅ No hay sesgo evidente hacia el centro")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_move_quality() 