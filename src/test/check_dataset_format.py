#!/usr/bin/env python3
"""
Script para verificar el formato del dataset de entrenamiento.
"""

import json
from pathlib import Path

def check_dataset_format():
    """Verifica el formato del dataset de entrenamiento"""
    print("🔍 VERIFICANDO FORMATO DEL DATASET")
    print("=" * 50)
    
    # Buscar archivos de dataset
    dataset_dir = Path("datasets")
    if not dataset_dir.exists():
        print("❌ Directorio 'datasets' no encontrado")
        return
    
    # Buscar archivos minimax
    minimax_files = list(dataset_dir.glob("tictactoe_minimax_*.jsonl"))
    if not minimax_files:
        print("❌ No se encontraron archivos minimax")
        return
    
    print(f"📁 Encontrados {len(minimax_files)} archivos minimax:")
    for file in minimax_files:
        print(f"   - {file.name}")
    
    # Analizar el archivo más reciente
    latest_file = max(minimax_files, key=lambda x: x.stat().st_mtime)
    print(f"\n📋 Analizando: {latest_file.name}")
    
    # Leer primeras líneas
    with open(latest_file, 'r', encoding='utf-8') as f:
        lines = [f.readline().strip() for _ in range(5)]
    
    print(f"\n📝 Primeras 5 líneas del dataset:")
    for i, line in enumerate(lines, 1):
        if line:
            try:
                data = json.loads(line)
                text = data.get("text", "")
                print(f"\n--- Línea {i} ---")
                print(text)
                
                # Analizar estructura
                if "<|move|>" in text:
                    move_start = text.find("<|move|>")
                    move_end = text.find("<|end|>", move_start)
                    if move_end != -1:
                        move_text = text[move_start:move_end + 6]
                        print(f"   Movimiento encontrado: '{move_text}'")
                        
                        # Verificar formato
                        import re
                        if re.search(r'<\|move\|><\|\d-\d\|><\|end\|>', move_text):
                            print("   ✅ Formato correcto")
                        else:
                            print("   ❌ Formato incorrecto")
                    else:
                        print("   ⚠️ <|move|> sin <|end|>")
                else:
                    print("   ❌ No contiene <|move|>")
                    
            except json.JSONDecodeError:
                print(f"   ❌ Error JSON en línea {i}")
    
    # Buscar archivos GRPO generados
    grpo_files = list(dataset_dir.glob("tictactoe_grpo_from_minimax_*.jsonl"))
    if grpo_files:
        print(f"\n📁 Encontrados {len(grpo_files)} archivos GRPO:")
        latest_grpo = max(grpo_files, key=lambda x: x.stat().st_mtime)
        print(f"   - {latest_grpo.name}")
        
        # Analizar formato GRPO
        with open(latest_grpo, 'r', encoding='utf-8') as f:
            grpo_line = f.readline().strip()
            if grpo_line:
                try:
                    data = json.loads(grpo_line)
                    print(f"\n📝 Formato GRPO generado:")
                    print(f"   Prompt: {data['prompt'][:100]}...")
                    print(f"   Completion: {data['completion']}")
                except json.JSONDecodeError:
                    print("   ❌ Error JSON en archivo GRPO")

if __name__ == "__main__":
    check_dataset_format() 