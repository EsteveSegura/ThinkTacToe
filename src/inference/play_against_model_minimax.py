#!/usr/bin/env python3
"""
Script interactivo para jugar contra el modelo de Tic-Tac-Toe en el terminal.
Usa el formato minimax y permite al usuario hacer movimientos.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import os
from pathlib import Path
import re

# Añadir el directorio raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.dataset.board_engine import (
    create_empty_board,
    apply_move,
    get_valid_moves,
    check_winner,
    is_draw,
    next_player
)
from src.dataset.board_tokenizer import board_to_token_representation

def print_board_ascii(board):
    """
    Imprime una visualización ASCII del tablero.
    """
    print("\n" + "="*50)
    print("           TIC-TAC-TOE")
    print("="*50)
    print("    0   1   2  (columnas)")
    print("  ┌───┬───┬───┐")
    for i in range(3):
        row_str = f"{i} │"
        for j in range(3):
            cell = board[i][j]
            if cell is None:
                row_str += "   │"
            else:
                row_str += f" {cell} │"
        print(row_str)
        if i < 2:
            print("  ├───┼───┼───┤")
    print("  └───┴───┴───┘")
    print("(filas)")
    print("="*50)

def get_user_move(board):
    """
    Obtiene el movimiento del usuario desde la terminal.
    """
    valid_moves = get_valid_moves(board)
    
    while True:
        try:
            print(f"\n🎮 Tu turno! Movimientos válidos: {valid_moves}")
            move_input = input("📝 Ingresa tu movimiento (fila,columna): ").strip()
            
            if move_input.lower() == 'quit':
                return None
            
            # Parsear entrada
            if ',' in move_input:
                row, col = map(int, move_input.split(','))
            else:
                # Si solo ingresa un número, asumir que es fila,columna
                move = int(move_input)
                row, col = move // 3, move % 3
            
            if (row, col) in valid_moves:
                return (row, col)
            else:
                print("❌ Movimiento inválido. Intenta de nuevo.")
                
        except (ValueError, IndexError):
            print("❌ Formato inválido. Usa: fila,columna (ej: 1,1)")

def create_prompt_minimax(board, current_player):
    """
    Crea un prompt en formato minimax para el modelo.
    """
    # Convertir el tablero al formato de tokens
    board_str = board_to_token_representation(board)
    
    # Determinar si es turno del bot o del jugador
    turn = "bot" if current_player == "X" else "player"
    
    prompt = f"{board_str}\n<|turn|>{turn}\n<|symbol|>{current_player}\n<|move|>"
    
    return prompt

def get_model_move(model, tokenizer, board, current_player):
    """
    Obtiene el movimiento del modelo.
    """
    prompt = create_prompt_minimax(board, current_player)
    
    # Tokenizar
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Mover inputs al mismo dispositivo que el modelo
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generar
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decodificar
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extraer el movimiento
    move_match = re.search(r'<\|move\|><\|(\d)-(\d)\|>', generated_text)
    if move_match:
        row, col = int(move_match.group(1)), int(move_match.group(2))
        return (row, col)
    else:
        # Fallback: elegir un movimiento aleatorio válido
        valid_moves = get_valid_moves(board)
        if valid_moves:
            return random.choice(valid_moves)
        return None

def main():
    print("🎯 TIC-TAC-TOE - Juega contra el modelo!")
    print("📋 Instrucciones:")
    print("   - Escribe 'quit' para salir")
    print("   - Usa formato: fila,columna (ej: 1,1)")
    print("   - Tú eres 'O', el modelo es 'X'")
    print("   - El modelo juega primero")
    
    # Cargar modelo
    model_path = "./qwen2.5-0.5b-tictactoe-sft-nothink-minmax/checkpoint-78"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encontró el modelo en {model_path}")
        return
    
    print(f"\n🤖 Cargando modelo desde: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✅ Modelo cargado exitosamente!")
    
    # Inicializar juego
    board = create_empty_board()
    turn_count = 0
    
    print("\n🎮 ¡Comienza el juego!")
    
    while True:
        turn_count += 1
        current_player = next_player(board)
        
        print_board_ascii(board)
        
        if current_player == "X":  # Turno del modelo
            print(f"\n🤖 Turno {turn_count}: El modelo está pensando...")
            move = get_model_move(model, tokenizer, board, current_player)
            
            if move is None:
                print("❌ Error: El modelo no pudo generar un movimiento válido")
                break
                
            print(f"🤖 El modelo juega en: {move}")
            board = apply_move(board, current_player, move)
            
        else:  # Turno del usuario
            print(f"\n👤 Turno {turn_count}: Tu turno!")
            move = get_user_move(board)
            
            if move is None:
                print("👋 ¡Gracias por jugar!")
                break
                
            board = apply_move(board, current_player, move)
        
        # Verificar ganador
        winner = check_winner(board)
        if winner:
            print_board_ascii(board)
            if winner == "X":
                print("🤖 ¡El modelo ha ganado!")
            else:
                print("👤 ¡Has ganado!")
            print(f"📊 La partida terminó en {turn_count} turnos")
            break
        
        # Verificar empate
        if is_draw(board):
            print_board_ascii(board)
            print("🤝 ¡EMPATE!")
            print(f"📊 La partida terminó en {turn_count} turnos")
            break
    
    # Preguntar si quiere jugar otra vez
    play_again = input("\n🔄 ¿Quieres jugar otra vez? (s/n): ").strip().lower()
    if play_again in ['s', 'si', 'sí', 'y', 'yes']:
        print("\n" + "="*50)
        main()
    else:
        print("👋 ¡Hasta luego!")

if __name__ == "__main__":
    import random
    main() 