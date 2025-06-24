#!/usr/bin/env python3
"""
Generador de dataset GRPO basado en el dataset minimax existente.
Convierte el formato minimax al formato GRPO con prompts y completions.
"""

import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def convert_minimax_to_grpo_format(minimax_line):
    """
    Convierte una línea del dataset minimax al formato GRPO.
    """
    try:
        # Parsear la línea minimax
        data = json.loads(minimax_line)
        text = data["text"]
        
        # Extraer el tablero y la información del jugador
        lines = text.split('\n')
        
        # Encontrar las líneas del tablero
        board_lines = []
        player_info = None
        move_info = None
        
        for line in lines:
            if line.startswith('<|board_start|>'):
                continue
            elif line.startswith('<|board_end|>'):
                continue
            elif line.startswith('<|turn|>'):
                player_info = line
            elif line.startswith('<|symbol|>'):
                continue
            elif line.startswith('<|move|>'):
                move_info = line
            elif line.startswith('<|end|>'):
                continue
            else:
                # Es una línea del tablero
                board_lines.append(line)
        
        # Construir el prompt (tablero + información del jugador)
        prompt_lines = ['<|board_start|>']
        prompt_lines.extend(board_lines)
        prompt_lines.append('<|board_end|>')
        prompt_lines.append(player_info)
        
        prompt = '\n'.join(prompt_lines)
        
        # Construir la completion (solo el movimiento, sin pensamiento)
        # Eliminar <|end|> duplicado si existe
        if move_info.endswith('<|end|>'):
            completion = move_info
        else:
            completion = f"{move_info}<|end|>"
        
        return {
            "prompt": prompt,
            "completion": completion
        }
        
    except Exception as e:
        print(f"Error procesando línea: {e}")
        return None

def generate_grpo_dataset_from_minimax():
    """
    Genera el dataset GRPO a partir del dataset minimax existente.
    """
    # Usar el archivo especificado por el usuario
    minimax_path = Path("datasets/tictactor_sft_nothink_minmax.jsonl")
    if not minimax_path.exists():
        print(f"❌ No se encontró el archivo {minimax_path}")
        return
    print(f"📁 Usando dataset minimax: {minimax_path}")
    
    # Generar nombre para el dataset GRPO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grpo_filename = f"tictactoe_grpo_from_minimax_{timestamp}.jsonl"
    grpo_path = Path("datasets") / grpo_filename
    
    print(f"🎯 Generando dataset GRPO: {grpo_path}")
    
    # Procesar el dataset minimax
    converted_count = 0
    error_count = 0
    
    with open(minimax_path, 'r', encoding='utf-8') as infile, \
         open(grpo_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            # Convertir al formato GRPO
            grpo_data = convert_minimax_to_grpo_format(line)
            
            if grpo_data:
                # Escribir en formato JSONL
                json.dump(grpo_data, outfile, ensure_ascii=False)
                outfile.write('\n')
                converted_count += 1
            else:
                error_count += 1
                print(f"⚠️ Error en línea {line_num}")
    
    print(f"\n✅ Dataset GRPO generado exitosamente!")
    print(f"📊 Estadísticas:")
    print(f"   - Líneas procesadas: {converted_count + error_count}")
    print(f"   - Conversiones exitosas: {converted_count}")
    print(f"   - Errores: {error_count}")
    print(f"   - Archivo generado: {grpo_path}")
    
    # Mostrar ejemplo del formato generado
    print(f"\n📝 Ejemplo del formato GRPO generado:")
    with open(grpo_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line:
            example = json.loads(first_line)
            print(f"Prompt: {example['prompt'][:100]}...")
            print(f"Completion: {example['completion']}")

if __name__ == "__main__":
    generate_grpo_dataset_from_minimax() 