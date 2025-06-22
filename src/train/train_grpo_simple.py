# train_grpo_simple.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

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
)

trainer = GRPOTrainer(
    model="GiRLaZo/qwen2.5-0.5b-tictactoe-sft-llm",
    reward_funcs=reward_think_length,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
