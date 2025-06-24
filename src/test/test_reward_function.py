#!/usr/bin/env python3
"""
Script de test para verificar la función de recompensa del entrenamiento GRPO.
"""

import sys
import re
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
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
        r"<\|final_answer\|>",
        r"<document>",
        r"</document>",
        r"<\|document\|>",
        r"</\|document\|>"
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
    # Corregido para solo permitir coordenadas válidas (0-2)
    match = re.search(r"<\|move\|><\|([0-2])-([0-2])\|><\|end\|>", text)
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

def get_current_player_from_prompt(prompt):
    """Extrae el jugador actual del prompt - CORREGIDO"""
    # Buscar el símbolo del jugador actual
    m = re.search(r'<\|symbol\|>([XO])', prompt)
    return m.group(1) if m else None

def evaluate_move_quality_minimax(board, move, current_player):
    """Evalúa la calidad del movimiento comparándolo con el movimiento óptimo de minimax"""
    if not board or not move:
        return 0.0
    
    row, col = move
    
    # Verificar si el movimiento es válido
    valid_moves = get_valid_moves(board)
    if move not in valid_moves:
        return -4.0  # Movimiento inválido - penalización más fuerte
    
    # Aplicar el movimiento
    new_board = apply_move(board, current_player, move)
    
    # Verificar si es un movimiento ganador
    if check_winner(new_board) == current_player:
        return 2.0  # Movimiento ganador - recompensa máxima
    
    # Verificar si bloquea un movimiento ganador del oponente
    # CORREGIDO: Verificar en el tablero original, no después de aplicar nuestro movimiento
    opponent = 'O' if current_player == 'X' else 'X'
    for valid_move in valid_moves:
        opponent_board = apply_move(board, opponent, valid_move)
        if check_winner(opponent_board) == opponent:
            # Si el oponente puede ganar con este movimiento, nuestro movimiento debe bloquearlo
            if move == valid_move:
                return 1.5  # Movimiento bloqueador - recompensa alta
    
    # Evaluación estratégica para movimientos no ganadores/no bloqueadores
    score = 0.0
    
    # Recompensar movimientos en el centro (estratégicamente importantes)
    if move == (1, 1):
        score += 0.8
    
    # Recompensar movimientos en esquinas (buenas posiciones estratégicas)
    elif move in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        score += 0.6
    
    # Recompensar movimientos que crean amenazas (dos en línea)
    threat_created = False
    for direction in [(0, 1), (1, 0), (1, 1), (1, -1)]:  # horizontal, vertical, diagonales
        count = 1  # Contar nuestro movimiento
        # Contar en dirección positiva
        for step in range(1, 3):
            new_row, new_col = row + step * direction[0], col + step * direction[1]
            if 0 <= new_row < 3 and 0 <= new_col < 3 and new_board[new_row][new_col] == current_player:
                count += 1
            else:
                break
        # Contar en dirección negativa
        for step in range(1, 3):
            new_row, new_col = row - step * direction[0], col - step * direction[1]
            if 0 <= new_row < 3 and 0 <= new_col < 3 and new_board[new_row][new_col] == current_player:
                count += 1
            else:
                break
        
        if count >= 2:
            threat_created = True
            break
    
    if threat_created:
        score += 0.4
    
    # Recompensar movimientos que bloquean amenazas del oponente
    opponent_threats_blocked = 0
    for direction in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        count = 0
        for step in range(-2, 3):
            new_row, new_col = row + step * direction[0], col + step * direction[1]
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                if board[new_row][new_col] == opponent:
                    count += 1
                elif board[new_row][new_col] == current_player:
                    count = 0  # Reset si encontramos nuestra pieza
                    break
        if count >= 2:
            opponent_threats_blocked += 1
    
    if opponent_threats_blocked > 0:
        score += 0.3 * opponent_threats_blocked
    
    # Recompensa base para movimientos válidos
    base_reward = 0.2
    
    return base_reward + score

def evaluate_format_quality(completion):
    """Evalúa la calidad del formato de la completion"""
    if not completion:
        return -1.0
    
    score = 0.0
    
    # Verificar formato básico del movimiento - CORREGIDO para coordenadas válidas
    if re.fullmatch(r"<\|move\|><\|[0-2]-[0-2]\|><\|end\|>", completion.strip()):
        score += 1.0  # Formato perfecto
    elif re.search(r"<\|move\|><\|[0-2]-[0-2]\|>", completion):
        score += 0.5  # Formato parcialmente correcto
    else:
        score -= 1.0  # Formato incorrecto
    
    # Penalizar texto extra
    if len(completion.strip()) > 50:  # Debería ser muy corto
        score -= 0.5
    
    return score

def reward_func(completions, prompts=None, **kwargs):
    """Función de recompensa optimizada para el formato minimax"""
    rewards = []
    
    for i, completion in enumerate(completions):
        prompt = prompts[i] if prompts else ""
        reward = 0.0
        
        # PENALIZACIÓN FUERTE por texto fuera de contexto
        if contains_bad_token(completion):
            reward = -3.0
            rewards.append(reward)
            continue
        
        # Extraer el movimiento del texto generado
        move = extract_move(completion)
        
        if not move:
            reward = -2.0  # No se puede extraer movimiento
            rewards.append(reward)
            continue
        
        # Extraer el tablero y jugador del prompt
        board = extract_board_from_prompt(prompt)
        current_player = get_current_player_from_prompt(prompt)
        
        if not board or not current_player:
            reward = -1.0  # No se puede extraer información del prompt
            rewards.append(reward)
            continue
        
        # Evaluar la calidad del movimiento
        move_quality = evaluate_move_quality_minimax(board, move, current_player)
        
        # Evaluar la calidad del formato
        format_quality = evaluate_format_quality(completion)
        
        # Combinar recompensas - Ajustado para mantener simetría
        final_reward = move_quality + format_quality * 0.5  # Reducir peso del formato
        rewards.append(final_reward)
    
    # Asegurar que tenemos el mismo número de recompensas que completions
    assert len(rewards) == len(completions), f"Recompensas: {len(rewards)}, Completions: {len(completions)}"
    
    return rewards

def print_board(board):
    """Imprime el tablero de forma legible"""
    for i, row in enumerate(board):
        row_str = []
        for j, cell in enumerate(row):
            if cell is None:
                row_str.append(" ")
            else:
                row_str.append(cell)
        print(f"  {' | '.join(row_str)}")
        if i < 2:
            print("  ---------")

def test_reward_function():
    """Función principal de test"""
    print("🧪 TESTEANDO FUNCIÓN DE RECOMPENSA GRPO")
    print("=" * 50)
    
    # Casos de prueba
    test_cases = [
        {
            "name": "Movimiento ganador X",
            "prompt": "<|board_start|>\n<|0-0|><|X|> <|0-1|><|X|> <|0-2|><|blank|>\n<|1-0|><|blank|> <|1-1|><|O|> <|1-2|><|blank|>\n<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>\n<|board_end|>\n<|turn|>player\n<|symbol|>X\n",
            "completion": "<|move|><|0-2|><|end|>",
            "expected_high": True
        },
        {
            "name": "Movimiento bloqueador O",
            "prompt": "<|board_start|>\n<|0-0|><|X|> <|0-1|><|X|> <|0-2|><|blank|>\n<|1-0|><|blank|> <|1-1|><|O|> <|1-2|><|blank|>\n<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>\n<|board_end|>\n<|turn|>player\n<|symbol|>O\n",
            "completion": "<|move|><|0-2|><|end|>",
            "expected_high": True
        },
        {
            "name": "Movimiento en centro (estratégico)",
            "prompt": "<|board_start|>\n<|0-0|><|blank|> <|0-1|><|blank|> <|0-2|><|blank|>\n<|1-0|><|blank|> <|1-1|><|blank|> <|1-2|><|blank|>\n<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>\n<|board_end|>\n<|turn|>player\n<|symbol|>X\n",
            "completion": "<|move|><|1-1|><|end|>",
            "expected_high": True
        },
        {
            "name": "Movimiento inválido (celda ocupada)",
            "prompt": "<|board_start|>\n<|0-0|><|X|> <|0-1|><|blank|> <|0-2|><|blank|>\n<|1-0|><|blank|> <|1-1|><|blank|> <|1-2|><|blank|>\n<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>\n<|board_end|>\n<|turn|>player\n<|symbol|>O\n",
            "completion": "<|move|><|0-0|><|end|>",
            "expected_high": False
        },
        {
            "name": "Formato incorrecto",
            "prompt": "<|board_start|>\n<|0-0|><|blank|> <|0-1|><|blank|> <|0-2|><|blank|>\n<|1-0|><|blank|> <|1-1|><|blank|> <|1-2|><|blank|>\n<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>\n<|board_end|>\n<|turn|>player\n<|symbol|>X\n",
            "completion": "movimiento 0-0",
            "expected_high": False
        },
        {
            "name": "Texto con tokens malos",
            "prompt": "<|board_start|>\n<|0-0|><|blank|> <|0-1|><|blank|> <|0-2|><|blank|>\n<|1-0|><|blank|> <|1-1|><|blank|> <|1-2|><|blank|>\n<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>\n<|board_end|>\n<|turn|>player\n<|symbol|>X\n",
            "completion": "<|move|><|0-0|><|end|><|thought|>pensando...<|end|>",
            "expected_high": False
        },
        {
            "name": "Movimiento en esquina (buena estrategia)",
            "prompt": "<|board_start|>\n<|0-0|><|blank|> <|0-1|><|blank|> <|0-2|><|blank|>\n<|1-0|><|blank|> <|1-1|><|blank|> <|1-2|><|blank|>\n<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>\n<|board_end|>\n<|turn|>player\n<|symbol|>X\n",
            "completion": "<|move|><|0-0|><|end|>",
            "expected_high": True
        }
    ]
    
    total_tests = len(test_cases)
    passed_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}/{total_tests}: {test_case['name']}")
        print("-" * 40)
        
        # Extraer información del prompt para mostrar
        board = extract_board_from_prompt(test_case['prompt'])
        current_player = get_current_player_from_prompt(test_case['prompt'])
        move = extract_move(test_case['completion'])
        
        print(f"Tablero actual:")
        if board:
            print_board(board)
        print(f"Jugador actual: {current_player}")
        print(f"Movimiento propuesto: {move}")
        print(f"Completion: '{test_case['completion']}'")
        
        # Calcular recompensa
        rewards = reward_func([test_case['completion']], [test_case['prompt']])
        reward = rewards[0]
        
        print(f"Recompensa obtenida: {reward:.3f}")
        
        # Evaluar resultado
        if test_case['expected_high']:
            if reward > 0.5:
                print("✅ PASÓ - Recompensa alta como esperado")
                passed_tests += 1
            else:
                print("❌ FALLÓ - Se esperaba recompensa alta")
        else:
            if reward < 0:
                print("✅ PASÓ - Recompensa baja/negativa como esperado")
                passed_tests += 1
            else:
                print("❌ FALLÓ - Se esperaba recompensa baja/negativa")
        
        # Mostrar desglose
        if move and board and current_player:
            move_quality = evaluate_move_quality_minimax(board, move, current_player)
            format_quality = evaluate_format_quality(test_case['completion'])
            print(f"  - Calidad del movimiento: {move_quality:.3f}")
            print(f"  - Calidad del formato: {format_quality:.3f}")
    
    print(f"\n📊 RESUMEN DE RESULTADOS")
    print("=" * 30)
    print(f"Tests pasados: {passed_tests}/{total_tests}")
    print(f"Porcentaje de éxito: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 ¡TODOS LOS TESTS PASARON!")
    else:
        print("⚠️ Algunos tests fallaron - revisar la función de recompensa")

if __name__ == "__main__":
    test_reward_function() 