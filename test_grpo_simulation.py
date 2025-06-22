#!/usr/bin/env python3
"""
Script para simular el comportamiento de GRPO y verificar la función de recompensa
"""

import sys
import re
from pathlib import Path

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

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
        reward = 0.0  # Recompensa base
        
        # PENALIZACIÓN FUERTE por texto fuera de contexto
        if contains_bad_token(completion):
            reward = -2.0
            rewards.append(reward)
            continue
        
        # Verificar formato básico
        think_count = completion.count("</player_think>")
        if think_count == 0:
            reward = -1.0  # Falta pensamiento
            rewards.append(reward)
            continue
        elif think_count > 1:
            reward -= 0.5  # Demasiados cierres
        
        # Verificar que después de </player_think> solo esté el movimiento
        if "</player_think>" in completion:
            after_think = completion.split("</player_think>")[-1].strip()
            if not re.fullmatch(r"<\|move\|><\|\d-\d\|><\|end\|>", after_think):
                reward -= 0.5  # Formato incorrecto después del pensamiento
        
        # Extraer el movimiento del texto generado
        move = extract_move(completion)
        
        if not move:
            reward = -1.0  # No se puede extraer movimiento
            rewards.append(reward)
            continue
        
        # Evaluar la calidad del pensamiento
        thinking_quality = evaluate_thinking_quality(completion)
        
        # BONUS por formato perfecto
        format_bonus = 0.1
        
        # Recompensa simple basada en el movimiento (simulada)
        move_quality = 0.5  # Simulamos una recompensa de movimiento
        
        final_reward = move_quality + thinking_quality + format_bonus + reward
        rewards.append(final_reward)
    
    # Asegurar que tenemos el mismo número de recompensas que completions
    assert len(rewards) == len(completions), f"Recompensas: {len(rewards)}, Completions: {len(completions)}"
    
    return rewards

def simulate_grpo_batch():
    """Simula un batch de GRPO con múltiples generaciones por prompt"""
    
    # Un prompt
    prompt = "<|board_start|>\n<|0-0|><|blank|> <|0-1|><|O|> <|0-2|><|X|>\n<|1-0|><|X|> <|1-1|><|blank|> <|1-2|><|blank|>\n<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|O|>\n<|board_end|>\n<|player|>X"
    
    # Múltiples completions generadas por GRPO (simuladas)
    completions = [
        # Completion 1 - Buena
        "<player_think>I need to analyze the board and make a strategic move. By placing an X at (1,1), I can control the center which is crucial for winning. This position allows me to create multiple threats and gives me flexibility for future moves. </player_think>\n<|move|><|1-1|><|end|>",
        
        # Completion 2 - Sin pensamiento
        "<|move|><|1-1|><|end|>",
        
        # Completion 3 - Con token malo
        "<|system|>Bad response</|system|><player_think>Good thinking</player_think>\n<|move|><|1-1|><|end|>",
        
        # Completion 4 - Movimiento inválido
        "<player_think>I will make a move</player_think>\n<|move|><|invalid|><|end|>",
        
        # Completion 5 - Muy buena
        "<player_think>After analyzing the current board state, I can see that placing an X at position (1,1) would be the optimal strategic move. The center position is crucial in Tic Tac Toe because it controls multiple potential winning lines - both diagonals, the middle row, and the middle column. By occupying this key position, I can set up future threats and limit my opponent's options. This move also disrupts any potential diagonal threats from Player O and gives me the most flexibility for subsequent moves. </player_think>\n<|move|><|1-1|><|end|>"
    ]
    
    # Crear prompts repetidos para simular el batch
    prompts = [prompt] * len(completions)
    
    print("=== SIMULACIÓN DE BATCH GRPO ===\n")
    print(f"Número de prompts: {len(prompts)}")
    print(f"Número de completions: {len(completions)}")
    print()
    
    try:
        rewards = reward_func(completions, prompts)
        print("✅ Función de recompensa ejecutada exitosamente")
        print(f"Número de recompensas: {len(rewards)}")
        print()
        
        print("Recompensas por completion:")
        for i, (completion, reward) in enumerate(zip(completions, rewards)):
            print(f"  Completion {i+1}: {reward:.3f}")
            print(f"    Texto: {completion[:80]}...")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simulate_grpo_batch() 