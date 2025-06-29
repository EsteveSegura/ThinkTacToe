#!/usr/bin/env python3
"""
Entrenamiento GRPO "ligero" para Tic-Tac-Toe (formato minimax).
---------------------------------------------------------------
• Busca automáticamente el dataset GRPO más reciente.
• Incluye función de recompensa + whitening interno (media 0, var 1).
• Callback de validación: cada N pasos genera jugadas y muestra % válidas.
• Se evitan flags de TRL no soportados por la versión instalada.
Uso rápido:
    python3 src/train/train_grpo_minimax_light.py --max_steps 500
"""
import json, re, sys, os, glob, argparse, random
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOTrainer, GRPOConfig

# --- Utilidades mínimas de tablero ------------------------------------------------
BLANK = None

def create_empty_board():
    return [[BLANK for _ in range(3)] for _ in range(3)]

def apply_move(board, player, move):
    row, col = move
    new_b = [r[:] for r in board]
    new_b[row][col] = player
    return new_b

def get_valid_moves(board):
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] is BLANK]

def check_winner(board):
    lines = []
    lines.extend(board)  # filas
    lines.extend([[board[r][c] for r in range(3)] for c in range(3)])  # columnas
    lines.append([board[i][i] for i in range(3)])
    lines.append([board[i][2 - i] for i in range(3)])
    for line in lines:
        if line[0] and all(cell == line[0] for cell in line):
            return line[0]
    return None
# -------------------------------------------------------------------------------

# --- Dataset helper --------------------------------------------------------------
DATASETS_DIR = Path("datasets")
LATEST_PATTERN = "tictactoe_grpo_from_minimax_*.jsonl"

def latest_dataset() -> Path:
    files = sorted(DATASETS_DIR.glob(LATEST_PATTERN))
    if not files:
        sys.exit("❌ No se encontró ningún dataset GRPO en 'datasets/'. Genera uno primero.")
    return files[-1]
# -------------------------------------------------------------------------------

# --- Parsing args ----------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="./datasets/tictactoe_grpo_from_minimax_20250629_022319.jsonl", help="Ruta al dataset GRPO jsonl")
parser.add_argument("--model", type=str, default="qwen2.5-0.5b-tictactoe-sft-nothink-minmax/checkpoint-156", help="Modelo base")
parser.add_argument("--output_dir", type=str, default="light_grpo_run", help="Directorio de salida")
parser.add_argument("--max_steps", type=int, default=1000, help="Pasos máximo de entrenamiento")
parser.add_argument("--logging_steps", type=int, default=25)
parser.add_argument("--val_every", type=int, default=100, help="Pasos entre validaciones")
parser.add_argument("--save_steps", type=int, default=250, help="Guardar checkpoint cada N pasos")
args = parser.parse_args()

DATASET_PATH = Path(args.dataset) if args.dataset else latest_dataset()
print(f"📁 Dataset: {DATASET_PATH} ({DATASET_PATH.stat().st_size/1e6:.1f} MB)")

dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
print("✅ Ejemplos:", len(dataset))

# --- Reward function -------------------------------------------------------------
MOVE_RE = re.compile(r"<\|move\|><\|([0-2])-([0-2])\|><\|end\|>")

def extract_move(text):
    m = MOVE_RE.search(text)
    return (int(m.group(1)), int(m.group(2))) if m else None

def extract_board(prompt):
    try:
        start = prompt.index("<|board_start|>") + len("<|board_start|>")
        end = prompt.index("<|board_end|>")
        section = [l.strip() for l in prompt[start:end].strip().split("\n") if l.strip()]
        board = create_empty_board()
        cell_pat = re.compile(r"<\|(\d)-(\d)\|><\|([^|]+)\|>")
        for line in section[:3]:
            for r,c,val in cell_pat.findall(line):
                if val in ("X","O"):
                    board[int(r)][int(c)] = val
        return board
    except ValueError:
        return None

def current_player(prompt):
    m = re.search(r"<\|symbol\|>([XO])", prompt)
    return m.group(1) if m else None

# Movimiento simple: +1 válido, -2 inválido, +2 ganador, +1.5 bloquea victoria

def evaluate_move(board, move, player):
    if move not in get_valid_moves(board):
        return -2.0
    after = apply_move(board, player, move)
    if check_winner(after) == player:
        return 2.0
    # bloquear
    opp = "O" if player == "X" else "X"
    for v in get_valid_moves(board):
        if check_winner(apply_move(board, opp, v)) == opp and v == move:
            return 1.5
    return 1.0


def reward_func(completions, prompts=None, **kwargs):
    rewards = []
    for comp, prompt in zip(completions, prompts):
        move = extract_move("<|move|><|" + comp if not comp.startswith("<|move|>") else comp)
        board = extract_board(prompt)
        player = current_player(prompt)
        if not move or board is None or player is None:
            rewards.append(-3.0)
            continue
        rewards.append(evaluate_move(board, move, player))
    # whitening (media 0, var 1) para evitar colapso
    arr = np.array(rewards, dtype=np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr.tolist()
# -------------------------------------------------------------------------------

# --- Configuración mínima --------------------------------------------------------
train_args = GRPOConfig(
    output_dir=args.output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=3,
    logging_steps=args.logging_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=1,
    remove_unused_columns=False,
    max_prompt_length=256,
    num_generations=2,
    max_completion_length=20,
    scale_rewards=False,  # ya hacemos whitening manual
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=100,
    bf16=True,
    dataloader_num_workers=0,
    max_steps=args.max_steps,
)

# --- Validación ligera -----------------------------------------------------------
class SimpleValidationCallback(TrainerCallback):
    def __init__(self, tokenizer, sample_prompts, every):
        self.tok = tokenizer; self.prompts = sample_prompts; self.every = every
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step==0 or state.global_step % self.every:
            return control
        model = kwargs['model']
        model.eval()
        inputs = self.tok(self.prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outs = model.generate(**inputs, max_new_tokens=20)
        txts = self.tok.batch_decode(outs, skip_special_tokens=True)
        total, ok = 0,0
        for p,t in zip(self.prompts, txts):
            mv = extract_move(t)
            brd = extract_board(p)
            pl = current_player(p)
            valid = mv and brd and mv in get_valid_moves(brd)
            ok += 1 if valid else 0
            total += 1
        print(f"\n🧪 Validación rápida @step {state.global_step}: {ok}/{total} movimientos válidos")
        model.train()
        return control
# -------------------------------------------------------------------------------

print("🔤 Cargando tokenizer…")
try:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
except Exception:
    tokenizer = None

trainer = GRPOTrainer(
    model=args.model,
    reward_funcs=reward_func,
    args=train_args,
    train_dataset=dataset,
)

if tokenizer:
    sample_prompts = [dataset[i]["prompt"] for i in random.sample(range(len(dataset)), k=5)]
    trainer.add_callback(SimpleValidationCallback(tokenizer, sample_prompts, args.val_every))

print("🚀 Comenzando entrenamiento…")
trainer.train()
print("💾 Guardando modelo final…")
trainer.save_model(train_args.output_dir)
if tokenizer:
    tokenizer.save_pretrained(train_args.output_dir)
print("✅ Entrenamiento finalizado. Modelo en", train_args.output_dir) 