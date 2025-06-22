# train_grpo_simple.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import logging
import os
import json
from datetime import datetime

# Configurar logging personalizado
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"grpo_training_{timestamp}.log")

# Crear archivo de log con formato personalizado
with open(log_file, 'w') as f:
    f.write("=== GRPO Training Logs ===\n")
    f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 50 + "\n\n")

# Configurar el logger estándar para otros mensajes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Solo mostrar en consola
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Iniciando entrenamiento GRPO - Logs guardados en: {log_file}")

# Cargar dataset local
dataset = load_dataset("json", data_files="./datasets/tictactoe_grpo_llm.jsonl", split="train")
logger.info(f"Dataset cargado: {len(dataset)} ejemplos")

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

# Configuración para entrenamiento rápido en H100
training_args = GRPOConfig(
    output_dir="qwen2.5-0.5b-tictactoe-grpo",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Batch pequeño para prueba rápida
    gradient_accumulation_steps=4,  # Acumular gradientes
    max_steps=100,  # Solo 100 pasos para prueba rápida
    learning_rate=1e-5,
    bf16=True,  # Usar bfloat16 para H100
    gradient_checkpointing=True,  # Ahorrar memoria
    save_steps=50,
    logging_steps=10,
    warmup_steps=10,
    # Configuración de logging
    logging_dir=f"logs/tensorboard_{timestamp}",
    report_to=["tensorboard"],
)

logger.info(f"Configuración de entrenamiento: {training_args}")

# Clase personalizada para logging de GRPO
class GRPOTrainerWithLogging(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = log_file
    
    def log(self, logs):
        # Llamar al método original
        super().log(logs)
        
        # Guardar en formato personalizado si es un step de logging
        if "step" in logs:
            step = logs["step"]
            if step % self.args.logging_steps == 0:
                with open(self.log_file, 'a') as f:
                    f.write(f"Step {step}: {json.dumps(logs, indent=2)}\n")
                    f.write("-" * 30 + "\n")

trainer = GRPOTrainerWithLogging(
    model="GiRLaZo/qwen2.5-0.5b-tictactoe-sft-llm",
    reward_funcs=reward_think_length,
    args=training_args,
    train_dataset=dataset,
)

logger.info("Iniciando entrenamiento...")
trainer.train()

# Escribir finalización en el log
with open(log_file, 'a') as f:
    f.write(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

logger.info("Entrenamiento completado!")
