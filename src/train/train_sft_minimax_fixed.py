#!/usr/bin/env python3
"""
Script de entrenamiento SFT corregido para el formato minimax.
"""

import os
import argparse
import torch
import json
from datetime import datetime
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model using SFT with minimax format')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-0.5B",
                      help='Name or path of the base model to use')
    parser.add_argument('--data_files', type=str, default="./datasets/tictactoe_minimax_20250624_205741.jsonl",
                      help='Path to the minimax dataset JSONL file')
    parser.add_argument('--output_dir', type=str, default="qwen2.5-0.5b-tictactoe-sft-minimax-fixed",
                      help='Directory to save the trained model')
    parser.add_argument('--logs_dir', type=str, default='./logs',
                      help='Directory to save training logs')
    parser.add_argument('--num_epochs', type=int, default=2,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='Learning rate')
    return parser.parse_args()

class LoggingCallback(TrainerCallback):
    """Callback personalizado para guardar logs de entrenamiento"""
    
    def __init__(self, logs_dir, model_type="sft_minimax"):
        super().__init__()
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

def validate_dataset_format(dataset):
    """Valida que el dataset tenga el formato correcto"""
    print("🔍 Validando formato del dataset...")
    
    # Verificar primeras líneas
    for i in range(min(3, len(dataset))):
        example = dataset[i]
        text = example["text"]
        
        print(f"\n--- Ejemplo {i+1} ---")
        print(f"Longitud: {len(text)} caracteres")
        
        # Verificar elementos clave
        checks = {
            "<|board_start|>": "<|board_start|>" in text,
            "<|board_end|>": "<|board_end|>" in text,
            "<|turn|>": "<|turn|>" in text,
            "<|symbol|>": "<|symbol|>" in text,
            "<|move|>": "<|move|>" in text,
            "<|end|>": "<|end|>" in text,
        }
        
        for key, found in checks.items():
            status = "✅" if found else "❌"
            print(f"   {status} {key}")
        
        # Verificar formato del movimiento
        if "<|move|>" in text and "<|end|>" in text:
            move_start = text.find("<|move|>")
            move_end = text.find("<|end|>", move_start)
            if move_end != -1:
                move_text = text[move_start:move_end + 6]
                print(f"   Movimiento: '{move_text}'")
                
                # Verificar patrón de coordenadas
                import re
                if re.search(r'<\|move\|><\|\d-\d\|><\|end\|>', move_text):
                    print("   ✅ Formato de movimiento correcto")
                else:
                    print("   ❌ Formato de movimiento incorrecto")
    
    print(f"\n📊 Dataset validado: {len(dataset)} ejemplos")

def main():
    args = parse_args()
    
    print("🚀 INICIANDO ENTRENAMIENTO SFT MINIMAX")
    print("=" * 50)
    print(f"Modelo base: {args.model_name}")
    print(f"Dataset: {args.data_files}")
    print(f"Directorio de salida: {args.output_dir}")
    print(f"Épocas: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 50)
    
    # Login a Hugging Face (opcional)
    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
        print("✅ Login a Hugging Face completado")
    
    # Verificar que el dataset existe
    if not os.path.exists(args.data_files):
        print(f"❌ Error: Dataset no encontrado: {args.data_files}")
        return
    
    # Cargar modelo y tokenizer
    print(f"\n📥 Cargando modelo base: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    # Configurar tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("🔧 Configurado pad_token = eos_token")
    
    model.resize_token_embeddings(len(tokenizer))
    print(f"📏 Vocabulario: {len(tokenizer)} tokens")

    # Habilitar gradient checkpointing para ahorrar memoria
    model.gradient_checkpointing_enable()
    print("💾 Gradient checkpointing habilitado")

    # Cargar dataset
    print(f"\n📁 Cargando dataset: {args.data_files}")
    dataset = load_dataset("json", data_files=args.data_files, split="train")
    print(f"✅ Dataset cargado: {len(dataset)} ejemplos")
    
    # Validar formato del dataset
    validate_dataset_format(dataset)

    def formatting_func(example):
        """Función de formateo para el dataset"""
        return example["text"]

    # Crear callback para logging
    logging_callback = LoggingCallback(args.logs_dir, "sft_minimax")

    # Configuración de entrenamiento optimizada
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        fp16=True,
        gradient_checkpointing=True,
        max_seq_length=512,  # Aumentado para el formato estructurado
        eos_token="<|end|>",  # CORREGIDO: usar el token correcto
        warmup_steps=50,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
    )

    print(f"\n⚙️ Configuración de entrenamiento:")
    print(f"   - Batch size: {training_args.per_device_train_batch_size}")
    print(f"   - Épocas: {training_args.num_train_epochs}")
    print(f"   - Learning rate: {training_args.learning_rate}")
    print(f"   - Max seq length: {training_args.max_seq_length}")
    print(f"   - EOS token: {training_args.eos_token}")

    # Trainer
    print(f"\n🤖 Inicializando trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        callbacks=[logging_callback],
    )

    print("🚀 Comenzando entrenamiento...")
    trainer.train()
    
    print("✅ Entrenamiento completado!")
    print(f"📁 Modelo guardado en: {args.output_dir}")

if __name__ == "__main__":
    main() 