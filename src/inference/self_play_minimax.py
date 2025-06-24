#!/usr/bin/env python3
"""
Script para que el modelo juegue contra sí mismo en una partida completa de Tic-Tac-Toe.
Usa el formato minimax y muestra una visualización ASCII del tablero.
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
    is_board_full
)
from src.dataset.board_tokenizer import board_to_token_representation

def print_board_ascii(board):
    """
    Imprime una visualización ASCII del tablero.
    """
    print("\n" + "="*30)
    print("TABLERO ACTUAL:")
    print("="*30)
    
    for i, row in enumerate(board):
        print("     |     |     ")
        print(f"  {row[0] or ' '}  |  {row[1] or ' '}  |  {row[2] or ' '}  ")
        if i < 2:
            print("_____|_____|_____")
        else:
            print("     |     |     ")
    
    print("="*30)
    print("Coordenadas: (fila, columna)")
    print("(0,0) | (0,1) | (0,2)")
    print("(1,0) | (1,1) | (1,2)")
    print("(2,0) | (2,1) | (2,2)")
    print("="*30 + "\n")

def parse_model_move(model_output: str) -> tuple:
    """
    Extrae las coordenadas del movimiento del modelo.
    """
    try:
        # Buscar patrón de coordenadas d-d en el texto
        match = re.search(r"(\d)-(\d)", model_output)
        if match:
            row, col = int(match.group(1)), int(match.group(2))
            return (row, col)
        return None
    except:
        return None

def create_prompt_minimax(board: list, player: str) -> str:
    """
    Crea el prompt para el modelo en formato minimax.
    """
    board_repr = board_to_token_representation(board)
    
    # Determinar si es turno del bot o del jugador
    if player == 'X':
        turn = 'bot'
    else:
        turn = 'player'
    
    return f"<|board_start|>\n{board_repr}\n<|board_end|>\n<|turn|>{turn}\n<|symbol|>{player}\n<|move|>"

def infer_move(model, tokenizer, board: list, player: str, device) -> tuple:
    """
    Realiza la inferencia para obtener un movimiento del modelo.
    """
    prompt = create_prompt_minimax(board, player)
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    eos_token = "<|end|>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token) if eos_token in tokenizer.get_vocab() else tokenizer.eos_token_id
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            eos_token_id=eos_token_id,
            temperature=0.1,  # Baja temperatura para movimientos más determinísticos
            do_sample=True,
            top_p=0.9,
        )
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    if "<|end|>" in output_text:
        output_text = output_text.split("<|end|>")[0] + "<|end|>"
    
    move = parse_model_move(output_text)
    return move, output_text

def self_play_game(model_path: str):
    """
    Ejecuta una partida completa donde el modelo juega contra sí mismo.
    """
    print("🎮 INICIANDO PARTIDA: MODELO vs MODELO")
    print("="*50)
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Cargar modelo y tokenizer
    try:
        print(f"Cargando modelo desde: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
        print("✅ Modelo cargado exitosamente")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return
    
    # Inicializar tablero
    board = create_empty_board()
    current_player = 'X'
    turn_count = 0
    
    print("\n🎯 La partida comienza con X (Bot)")
    print_board_ascii(board)
    
    # Bucle principal del juego
    while True:
        turn_count += 1
        print(f"🔄 TURNO {turn_count} - Jugador: {current_player}")
        print("-" * 30)
        
        # Obtener movimiento del modelo
        print(f"🤖 El modelo está pensando su movimiento...")
        move, model_output = infer_move(model, tokenizer, board, current_player, device)
        
        if move is None:
            print(f"❌ Error: El modelo no pudo generar un movimiento válido")
            print(f"Respuesta del modelo: {model_output}")
            break
        
        row, col = move
        
        # Verificar si el movimiento es válido
        valid_moves = get_valid_moves(board)
        if move not in valid_moves:
            print(f"❌ Error: Movimiento inválido ({row}, {col})")
            print(f"Movimientos válidos: {valid_moves}")
            print(f"Respuesta del modelo: {model_output}")
            break
        
        # Aplicar el movimiento
        board = apply_move(board, current_player, move)
        
        print(f"✅ Movimiento realizado: ({row}, {col}) por {current_player}")
        print(f"🤖 Respuesta del modelo: {model_output}")
        
        # Mostrar tablero actualizado
        print_board_ascii(board)
        
        # Verificar si hay un ganador
        winner = check_winner(board)
        if winner:
            print(f"🏆 ¡GANADOR: {winner}!")
            print(f"🎉 La partida terminó en {turn_count} turnos")
            break
        
        # Verificar si es empate
        if is_board_full(board):
            print(f"🤝 ¡EMPATE!")
            print(f"📊 La partida terminó en {turn_count} turnos")
            break
        
        # Cambiar jugador
        current_player = 'O' if current_player == 'X' else 'X'
        
        # Pausa para mejor visualización
        input("Presiona Enter para continuar...")
    
    print("\n🎮 FIN DE LA PARTIDA")
    print("="*50)

if __name__ == "__main__":
    # Ruta del modelo
    model_path = "./qwen2.5-0.5b-tictactoe-sft-nothink-minmax/checkpoint-78"
    
    # Verificar que el modelo existe
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encontró el modelo en {model_path}")
        print("Por favor, verifica la ruta del modelo.")
        sys.exit(1)
    
    # Ejecutar la partida
    self_play_game(model_path) 