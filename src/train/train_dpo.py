import argparse
import os
import json
from datetime import datetime
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model using DPO')
    parser.add_argument('--path', type=str, required=True,
                      help='Path to the SFT checkpoint to use as base model')
    parser.add_argument('--data_files', type=str, required=True,
                      help='Path to the DPO dataset JSONL file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the trained model')
    parser.add_argument('--logs_dir', type=str, default='./logs',
                      help='Directory to save training logs')
    return parser.parse_args()

class LoggingCallback:
    """Callback personalizado para guardar logs de entrenamiento"""
    
    def __init__(self, logs_dir, model_type="dpo"):
        self.logs_dir = logs_dir
        self.model_type = model_type
        os.makedirs(logs_dir, exist_ok=True)
        
        # Crear archivo de logs con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(logs_dir, f"{model_type}_training_logs_{timestamp}.txt")
        
        # Escribir header del archivo
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {model_type.upper()} Training Logs ===\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Método llamado cuando se generan logs durante el entrenamiento"""
        if logs is not None:
            # Guardar logs en archivo
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"Step {state.global_step}: {json.dumps(logs, indent=2)}\n")
                f.write("-" * 30 + "\n")
            
            # También imprimir en consola
            print(f"Step {state.global_step}: {logs}")

def main():
    args = parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(args.path)
    tokenizer = AutoTokenizer.from_pretrained(args.path)

    train_dataset = load_dataset("json", data_files=args.data_files, split="train")

    # Crear callback para logging
    logging_callback = LoggingCallback(args.logs_dir, "dpo")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        logging_steps=10
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        num_train_epochs=3,
        callbacks=[logging_callback]
    )

    trainer.train()

if __name__ == "__main__":
    main()