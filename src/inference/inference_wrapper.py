#!/usr/bin/env python3
"""
Wrapper de inferencia que convierte coordenadas simples al formato correcto.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import re

def load_model():
    """Carga el modelo y tokenizer"""
    model_name = "qwen2.5-0.5b-tictactoe-sft-nothink-minmax"
    
    # Buscar el último checkpoint
    checkpoint_dir = model_name
    if os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            model_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"Cargando checkpoint: {model_path}")
        else:
            model_path = checkpoint_dir
            print(f"No se encontraron checkpoints, usando: {model_path}")
    else:
        model_path = "GiRLaZo/qwen2.5-0.5b-tictactoe-sft-llm"
        print(f"Directorio de entrenamiento no encontrado, usando modelo base: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    
    return model, tokenizer, device

def generate_move(model, tokenizer, device, prompt, max_new_tokens=20):
    """Genera un movimiento con configuración optimizada"""
    
    # Tokenizar el prompt
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Configurar tokens de fin
    eos_token = "<|end|>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token) if eos_token in tokenizer.get_vocab() else tokenizer.eos_token_id
    
    # Configuración de generación más conservadora
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=0.1,  # Muy baja temperatura para respuestas deterministas
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,  # Penalizar repeticiones
            pad_token_id=eos_token_id,
            num_return_sequences=1,
        )
    
    # Decodificar solo los nuevos tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    
    return output_text

def extract_and_format_move(output_text):
    """Extrae coordenadas y las convierte al formato correcto"""
    
    # Buscar el patrón completo primero
    move_match = re.search(r'<\|move\|><\|(\d)-(\d)\|><\|end\|>', output_text)
    if move_match:
        row, col = int(move_match.group(1)), int(move_match.group(2))
        return (row, col), f"<|move|><|{row}-{col}|><|end|>", "formato_completo"
    
    # Buscar solo coordenadas si no encuentra el formato completo
    coord_match = re.search(r'(\d)-(\d)', output_text)
    if coord_match:
        row, col = int(coord_match.group(1)), int(coord_match.group(2))
        formatted_move = f"<|move|><|{row}-{col}|><|end|>"
        return (row, col), formatted_move, "coordenadas_simples"
    
    return None, None, "no_encontrado"

def infer_move(prompt):
    """Función principal de inferencia con post-procesamiento"""
    print("🤖 INFERENCIA CON POST-PROCESAMIENTO")
    print("=" * 50)
    
    # Cargar modelo
    model, tokenizer, device = load_model()
    
    print("📋 Prompt de entrada:")
    print(prompt)
    print("\n" + "="*50 + "\n")
    
    # Generar respuesta
    raw_output = generate_move(model, tokenizer, device, prompt)
    
    print("📤 Respuesta cruda del modelo:")
    print(f"'{raw_output}'")
    print("\n" + "="*50 + "\n")
    
    # Extraer y formatear movimiento
    move, formatted_move, format_type = extract_and_format_move(raw_output)
    
    if move:
        row, col = move
        print(f"🎯 Movimiento extraído: ({row}, {col})")
        print(f"📝 Formato detectado: {format_type}")
        print(f"✅ Formato corregido: {formatted_move}")
        
        if format_type == "formato_completo":
            print("🎉 El modelo generó el formato correcto directamente")
        elif format_type == "coordenadas_simples":
            print("🔧 Se aplicó post-procesamiento para corregir el formato")
            
        return (row, col), formatted_move
    else:
        print("❌ No se pudo extraer movimiento")
        print(f"Texto completo: '{raw_output}'")
        return None, None

def test_wrapper():
    """Función de test"""
    # Prompt de prueba
    test_prompt = """<|board_start|>
<|0-0|><|blank|> <|0-1|><|blank|> <|0-2|><|X|>
<|1-0|><|O|> <|1-1|><|blank|> <|1-2|><|blank|>
<|2-0|><|blank|> <|2-1|><|X|> <|2-2|><|blank|>
<|board_end|>
<|turn|>bot
<|symbol|>X
<|move|>"""
    
    move, formatted_move = infer_move(test_prompt)
    
    if move:
        print(f"\n🎮 Movimiento final: {move}")
        print(f"📋 Formato para dataset: {formatted_move}")

if __name__ == "__main__":
    test_wrapper() 