#!/usr/bin/env python3
"""
Script de prueba para verificar la función de recompensa GRPO
"""

import sys
import re
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.dataset.board_engine import (
    create_empty_board,
    apply_move,
    get_valid_moves,
    next_player,
    check_winner,
    is_draw
)

def contains_bad_token(text):
    """Verifica si el texto contiene tokens problemáticos o fuera de contexto"""
    bad_patterns = [
        r"<\|endoftext\|>",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"<\|system\|>",
        r"<\|user\|>",
        r"<\|assistant\|>",
        r"<\|function\|>",
        r"<\|function_results\|>",
        r"<\|function_calls\|>",
        r"<\|observation\|>",
        r"<\|thought\|>",
        r"<\|action\|>",
        r"<\|action_input\|>",
        r"<\|final_answer\|>"
    ]
    
    for pattern in bad_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Verificar si contiene múltiples movimientos
    move_count = len(re.findall(r"<\|move\|>", text))
    if move_count > 1:
        return True
    
    return False

def extract_move(text):
    """Extrae el movimiento del texto generado por el modelo"""
    match = re.search(r"<\|move\|><\|(\d)-(\d)\|><\|end\|>", text)
    return (int(match.group(1)), int(match.group(2))) if match else None

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
    """Evalúa la calidad del movimiento en el contexto del tablero"""
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
    
    # Verificar si es un movimiento estratégico (centro, esquinas)
    if move == (1, 1):  # Centro
        return 0.6
    elif move in [(0, 0), (0, 2), (2, 0), (2, 2)]:  # Esquinas
        return 0.4
    else:  # Laterales
        return 0.2

def evaluate_thinking_quality(completion):
    """Evalúa la calidad del pensamiento del modelo"""
    if not completion:
        return 0.0
    
    # Extraer el contenido del pensamiento
    think_match = re.search(r"<player_think>(.*?)</player_think>", completion, re.DOTALL)
    if not think_match:
        return 0.0
    
    thinking = think_match.group(1).strip()
    
    # Criterios de calidad del pensamiento
    score = 0.0
    
    # Longitud mínima (debe ser sustancial)
    if len(thinking) < 50:
        score -= 0.3
    elif len(thinking) > 200:
        score += 0.2
    
    # Debe contener análisis estratégico
    strategic_keywords = [
        'analyze', 'strategy', 'strategic', 'position', 'control', 'threat',
        'win', 'block', 'diagonal', 'row', 'column', 'center', 'corner',
        'opportunity', 'advantage', 'flexibility', 'multiple', 'future'
    ]
    
    keyword_count = sum(1 for keyword in strategic_keywords if keyword.lower() in thinking.lower())
    score += min(keyword_count * 0.1, 0.5)  # Máximo 0.5 por keywords
    
    # Debe mencionar el movimiento específico
    if re.search(r'\(\d,\d\)', thinking):
        score += 0.2
    
    # Debe explicar el razonamiento
    if any(word in thinking.lower() for word in ['because', 'since', 'therefore', 'thus', 'so']):
        score += 0.2
    
    return max(score, -0.5)  # No penalizar demasiado

def reward_func(completions, prompts=None, **kwargs):
    """Función de recompensa que evalúa la calidad de los movimientos y el pensamiento"""
    rewards = []
    
    for i, completion in enumerate(completions):
        prompt = prompts[i] if prompts else ""
        
        # PENALIZACIÓN FUERTE por texto fuera de contexto
        if contains_bad_token(completion):
            rewards.append(-2.0)
            continue
        
        # Verificar formato básico
        think_count = completion.count("</player_think>")
        if think_count == 0:
            rewards.append(-1.0)  # Falta pensamiento
            continue
        elif think_count > 1:
            rewards.append(-0.5)  # Demasiados cierres
        
        # Verificar que después de </player_think> solo esté el movimiento
        if "</player_think>" in completion:
            after_think = completion.split("</player_think>")[-1].strip()
            if not re.fullmatch(r"<\|move\|><\|\d-\d\|><\|end\|>", after_think):
                rewards.append(-0.5)  # Formato incorrecto después del pensamiento
        
        # Extraer el movimiento del texto generado
        move = extract_move(completion)
        
        if not move:
            rewards.append(-1.0)  # No se puede extraer movimiento
            continue
        
        # Extraer el tablero del prompt
        board = extract_board_from_prompt(prompt)
        
        # Evaluar la calidad del movimiento
        move_quality = evaluate_move_quality(board, move)
        
        # Evaluar la calidad del pensamiento
        thinking_quality = evaluate_thinking_quality(completion)
        
        # BONUS por formato perfecto
        format_bonus = 0.1
        
        final_reward = move_quality + thinking_quality + format_bonus
        rewards.append(final_reward)
    
    return rewards

def test_examples():
    """Prueba la función de recompensa con ejemplos del dataset"""
    
    # Ejemplos del dataset
    test_cases = [
        {
            "prompt": "<|board_start|>\n<|0-0|><|blank|> <|0-1|><|O|> <|0-2|><|X|>\n<|1-0|><|X|> <|1-1|><|blank|> <|1-2|><|blank|>\n<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|O|>\n<|board_end|>\n<|player|>X",
            "completion": "<player_think>By placing an \"X\" at the center (1,1), I aim to control a pivotal position that influences multiple potential winning lines—rows, columns, and diagonals. This strategic move not only strengthens my own possibilities for creating two separate threats in subsequent turns but also disrupts any potential diagonal or line formations that Player O might be planning. The center is crucial as it allows me to potentially set up future moves where I can force a win by creating multiple threats that Player O cannot block all at once. Additionally, occupying the center limits Player O's ability to expand their influence across the board effectively. This move positions me advantageously for the next steps in the game. </player_think>\n<|move|><|1-1|><|end|>"
        },
        {
            "prompt": "<|board_start|>\n<|0-0|><|blank|> <|0-1|><|O|> <|0-2|><|X|>\n<|1-0|><|O|> <|1-1|><|X|> <|1-2|><|O|>\n<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|X|>\n<|board_end|>\n<|player|>X",
            "completion": "<player_think>I need to make a strategic move. Looking at the board, placing my 'X' in the top-left corner (0,0) will complete the diagonal from (0,0) to (2,2), giving me three in a row and securing an immediate win. Additionally, this move blocks any potential threats Player O might have on that diagonal. It's a strong position as it not only ensures victory but also consolidates control over the board's key areas. </player_think>\n<|move|><|0-0|><|end|>"
        },
        {
            "prompt": "<|board_start|>\n<|0-0|><|O|> <|0-1|><|blank|> <|0-2|><|X|>\n<|1-0|><|X|> <|1-1|><|blank|> <|1-2|><|X|>\n<|2-0|><|blank|> <|2-1|><|O|> <|2-2|><|O|>\n<|board_end|>\n<|player|>X",
            "completion": "<player_think>I need to analyze the current board state and determine the best move as player \"X\". The goal is to either win the game or prevent the opponent from winning while setting up potential winning opportunities.\n\nThe current board state is:\n\n[\n  [\"O\", \"\", \"X\"],\n  [\"X\", \"\", \"X\"],\n  [\"\", \"O\", \"O\"]\n]\n\nLooking at this, I can see that my pieces (\"X\") are positioned in such a way that completing a row or column could lead to an immediate win. Specifically, if I place an \"X\" in the center position (1,1), it will complete the middle row as all three positions in the middle row will be occupied by \"X\". This would result in an immediate victory for me.\n\nAdditionally, placing an \"X\" in the center is a strategic move because the center is part of multiple potential winning lines: the middle row, the middle column, and both diagonals. By controlling the center, I not only secure this immediate win but also set myself up for future opportunities if the game were to continue.\n\nI should also consider whether the opponent has any potential threats that could result in an immediate win after my move. However, based on the current board state, it appears that the opponent does not have a clear path to completing a row, column, or diagonal in their next move. This makes it safe for me to proceed with the winning move without risking losing the game.\n\nTherefore, moving to position (1,1) is the optimal choice as it secures an immediate win while also maintaining control over key positions on the board. </player_think>\n<|move|><|1-1|><|end|>"
        }
    ]
    
    print("=== PRUEBA DE FUNCIÓN DE RECOMPENSA GRPO ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Ejemplo {i}:")
        print(f"Prompt: {test_case['prompt'][:100]}...")
        print(f"Completion: {test_case['completion'][:100]}...")
        
        # Calcular recompensa
        reward = reward_func([test_case['completion']], [test_case['prompt']])[0]
        
        # Análisis detallado
        print(f"Recompensa total: {reward:.3f}")
        
        # Desglose
        move = extract_move(test_case['completion'])
        board = extract_board_from_prompt(test_case['prompt'])
        move_quality = evaluate_move_quality(board, move)
        thinking_quality = evaluate_thinking_quality(test_case['completion'])
        
        print(f"  - Calidad del movimiento: {move_quality:.3f}")
        print(f"  - Calidad del pensamiento: {thinking_quality:.3f}")
        print(f"  - Bonus de formato: 0.100")
        print(f"  - Movimiento extraído: {move}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_examples() 