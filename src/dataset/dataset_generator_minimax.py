#!/usr/bin/env python3
"""
Generador de dataset para Tic-Tac-Toe con movimientos óptimos calculados por minimax.
Produce ejemplos en formato JSONL para entrenamiento de modelos.
"""

import json
import random
from typing import List, Tuple, Optional, Set
from itertools import product
import os
from datetime import datetime


class TicTacToeMinimax:
    """Implementación del juego Tic-Tac-Toe con algoritmo minimax para movimientos óptimos."""
    
    def __init__(self):
        self.board = [['' for _ in range(3)] for _ in range(3)]
        self.empty_cell = '<|blank|>'
        
    def reset_board(self):
        """Reinicia el tablero."""
        self.board = [['' for _ in range(3)] for _ in range(3)]
    
    def get_board_state(self) -> List[List[str]]:
        """Retorna el estado actual del tablero."""
        return [row[:] for row in self.board]
    
    def set_board_state(self, state: List[List[str]]):
        """Establece el estado del tablero."""
        self.board = [row[:] for row in state]
    
    def is_valid_move(self, row: int, col: int) -> bool:
        """Verifica si un movimiento es válido."""
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] == ''
    
    def make_move(self, row: int, col: int, player: str) -> bool:
        """Realiza un movimiento en el tablero."""
        if self.is_valid_move(row, col):
            self.board[row][col] = player
            return True
        return False
    
    def check_winner(self) -> Optional[str]:
        """Verifica si hay un ganador."""
        # Filas
        for row in self.board:
            if row[0] == row[1] == row[2] != '':
                return row[0]
        
        # Columnas
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != '':
                return self.board[0][col]
        
        # Diagonales
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != '':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != '':
            return self.board[0][2]
        
        return None
    
    def is_board_full(self) -> bool:
        """Verifica si el tablero está lleno."""
        return all(self.board[i][j] != '' for i, j in product(range(3), range(3)))
    
    def get_available_moves(self) -> List[Tuple[int, int]]:
        """Retorna las posiciones disponibles."""
        return [(i, j) for i, j in product(range(3), range(3)) if self.board[i][j] == '']
    
    def minimax(self, depth: int, alpha: float, beta: float, is_maximizing: bool) -> Tuple[float, Optional[Tuple[int, int]]]:
        """Algoritmo minimax con poda alpha-beta."""
        winner = self.check_winner()
        
        if winner == 'X':
            return 1.0, None
        elif winner == 'O':
            return -1.0, None
        elif self.is_board_full():
            return 0.0, None
        
        available_moves = self.get_available_moves()
        
        if is_maximizing:
            best_score = float('-inf')
            best_move = None
            
            for move in available_moves:
                self.board[move[0]][move[1]] = 'X'
                score, _ = self.minimax(depth + 1, alpha, beta, False)
                self.board[move[0]][move[1]] = ''
                
                if score > best_score:
                    best_score = score
                    best_move = move
                
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            
            return best_score, best_move
        else:
            best_score = float('inf')
            best_move = None
            
            for move in available_moves:
                self.board[move[0]][move[1]] = 'O'
                score, _ = self.minimax(depth + 1, alpha, beta, True)
                self.board[move[0]][move[1]] = ''
                
                if score < best_score:
                    best_score = score
                    best_move = move
                
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            
            return best_score, best_move
    
    def get_optimal_move(self, player: str) -> Tuple[int, int]:
        """Calcula el movimiento óptimo para un jugador."""
        if player == 'X':
            _, move = self.minimax(0, float('-inf'), float('inf'), True)
        else:
            _, move = self.minimax(0, float('-inf'), float('inf'), False)
        return move
    
    def count_pieces(self) -> Tuple[int, int]:
        """Cuenta el número de fichas X y O en el tablero."""
        x_count = sum(1 for i, j in product(range(3), range(3)) if self.board[i][j] == 'X')
        o_count = sum(1 for i, j in product(range(3), range(3)) if self.board[i][j] == 'O')
        return x_count, o_count
    
    def get_next_player(self) -> str:
        """Determina qué jugador debe mover basado en el estado actual."""
        x_count, o_count = self.count_pieces()
        return 'X' if x_count == o_count else 'O'
    
    def board_to_string(self) -> str:
        """Convierte el tablero al formato de string especificado."""
        lines = ['<|board_start|>']
        
        for i in range(3):
            row_parts = []
            for j in range(3):
                cell_content = self.board[i][j] if self.board[i][j] else 'blank'
                row_parts.append(f'<|{i}-{j}|><|{cell_content}|>')
            lines.append(' '.join(row_parts))
        
        lines.append('<|board_end|>')
        return '\n'.join(lines)


class MinimaxDatasetGenerator:
    """Generador de dataset con movimientos óptimos calculados por minimax."""
    
    def __init__(self):
        self.game = TicTacToeMinimax()
        self.generated_positions = set()
        
    def get_canonical_form(self, board: List[List[str]]) -> str:
        """Convierte una posición a su forma canónica para evitar duplicados por simetría."""
        # Implementación simple: usa la representación como string
        # En una implementación más avanzada, se aplicarían todas las simetrías
        return str(board)
    
    def generate_random_position(self, target_pieces: int) -> Optional[List[List[str]]]:
        """Genera una posición aleatoria con un número específico de fichas."""
        max_attempts = 1000
        
        for _ in range(max_attempts):
            self.game.reset_board()
            pieces_placed = 0
            players = ['X', 'O']
            current_player = 0
            
            while pieces_placed < target_pieces:
                available_moves = self.game.get_available_moves()
                if not available_moves:
                    break
                
                move = random.choice(available_moves)
                self.game.make_move(move[0], move[1], players[current_player])
                pieces_placed += 1
                current_player = (current_player + 1) % 2
            
            if pieces_placed == target_pieces:
                canonical = self.get_canonical_form(self.game.get_board_state())
                if canonical not in self.generated_positions:
                    self.generated_positions.add(canonical)
                    return self.game.get_board_state()
        
        return None
    
    def create_example(self, board: List[List[str]]) -> Optional[dict]:
        """Crea un ejemplo en el formato especificado."""
        self.game.set_board_state(board)
        
        # Verificar que la posición es válida
        winner = self.game.check_winner()
        if winner:
            return None  # No generar posiciones con ganador
        
        next_player = self.game.get_next_player()
        
        # Calcular movimiento óptimo
        optimal_move = self.game.get_optimal_move(next_player)
        if optimal_move is None:
            return None  # No hay movimientos disponibles
        
        # Crear el texto del ejemplo
        board_text = self.game.board_to_string()
        move_text = f'<|move|><|{optimal_move[0]}-{optimal_move[1]}|><|end|>'
        
        # Determinar si el próximo jugador es el bot o el jugador humano
        # Asumimos que X es el bot y O es el jugador humano
        if next_player == 'X':
            player_type = 'bot'
        else:
            player_type = 'player'
        
        example_text = f"{board_text}\n<|turn|>{player_type}\n<|symbol|>{next_player}\n{move_text}"
        
        return {"text": example_text}
    
    def generate_dataset(self, total_examples: int = 1000) -> List[dict]:
        """Genera el dataset completo."""
        examples = []
        
        # Distribución de piezas según especificaciones
        piece_distributions = {
            'opening': (0, 2, int(total_examples * 0.25)),      # 25% aperturas
            'midgame': (3, 5, int(total_examples * 0.50)),      # 50% medio juego
            'endgame': (6, 8, int(total_examples * 0.25))       # 25% final
        }
        
        for phase, (min_pieces, max_pieces, target_count) in piece_distributions.items():
            print(f"Generando {target_count} ejemplos para {phase} ({min_pieces}-{max_pieces} piezas)...")
            
            phase_examples = 0
            attempts = 0
            max_attempts = target_count * 10
            
            while phase_examples < target_count and attempts < max_attempts:
                target_pieces = random.randint(min_pieces, max_pieces)
                position = self.generate_random_position(target_pieces)
                
                if position:
                    example = self.create_example(position)
                    if example:
                        examples.append(example)
                        phase_examples += 1
                
                attempts += 1
            
            print(f"Generados {phase_examples} ejemplos para {phase}")
        
        return examples
    
    def save_dataset(self, examples: List[dict], filename: str):
        """Guarda el dataset en formato JSONL."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"Dataset guardado en: {filename}")
        print(f"Total de ejemplos: {len(examples)}")


def main():
    """Función principal para generar el dataset."""
    generator = MinimaxDatasetGenerator()
    
    # Generar dataset
    print("Generando dataset de Tic-Tac-Toe con movimientos óptimos...")
    examples = generator.generate_dataset(total_examples=1000)
    
    # Guardar dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"datasets/tictactoe_minimax_{timestamp}.jsonl"
    generator.save_dataset(examples, filename)
    
    print("¡Dataset generado exitosamente!")


if __name__ == "__main__":
    main() 