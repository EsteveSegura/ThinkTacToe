# train_grpo_simple.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import json
import os
import sys
from datetime import datetime

# Configurar logging simple
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_file = f"logs/grpo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Crear directorio logs si no existe
os.makedirs("logs", exist_ok=True)

# Clase para capturar la salida y guardarla en archivo
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirigir stdout al archivo
sys.stdout = Logger(log_file)

print(f"=== GRPO Training Logs ===")
print(f"Started at: {timestamp}")
print("=" * 50)
print()

# Cargar dataset local
dataset = load_dataset("json", data_files="./datasets/tictactoe_grpo_llm.jsonl", split="train")

# Función de recompensa simple: recompensa basada en la longitud del pensamiento
def reward_think_length(completions, **kwargs):
    """Recompensa basada en la longitud del pensamiento del jugador"""
    rewards = []
    for completion in completions:
        # Buscar la sección de pensamiento
        if "<player_think>" in completion and "</player_think>" in completion:
            think_start = completion.find("<player_think>") + len("<player_think>")
            think_end = completion.find("</player_think>")
            think_content = completion[think_start:think_end].strip()
            # Recompensa basada en la longitud del pensamiento (más largo = mejor)
            reward = min(len(think_content) / 100.0, 2.0)  # Normalizar y limitar
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards

# Configuración para entrenamiento completo en H100
training_args = GRPOConfig(
    output_dir="qwen2.5-0.5b-tictactoe-grpo",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Batch pequeño para prueba rápida
    gradient_accumulation_steps=4,  # Acumular gradientes
    learning_rate=1e-5,
    bf16=True,  # Usar bfloat16 para H100
    gradient_checkpointing=True,  # Ahorrar memoria
    save_steps=50,
    logging_steps=10,
    warmup_steps=10,
)

trainer = GRPOTrainer(
    model="GiRLaZo/qwen2.5-0.5b-tictactoe-sft-llm",
    reward_funcs=reward_think_length,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
