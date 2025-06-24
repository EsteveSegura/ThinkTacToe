from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Modelo entrenado con GRPO - cargar el último checkpoint
model_name = "qwen2.5-0.5b-tictactoe-sft-nothink-minmax"

# Buscar el último checkpoint
checkpoint_dir = model_name
if os.path.exists(checkpoint_dir):
    # Buscar la carpeta del checkpoint más reciente
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        # Ordenar por número de checkpoint y tomar el último
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        model_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Cargando checkpoint: {model_path}")
    else:
        # Si no hay checkpoints, usar el directorio base
        model_path = checkpoint_dir
        print(f"No se encontraron checkpoints, usando: {model_path}")
else:
    # Si no existe el directorio, usar el modelo base
    model_path = "GiRLaZo/qwen2.5-0.5b-tictactoe-sft-llm"
    print(f"Directorio de entrenamiento no encontrado, usando modelo base: {model_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()

def infer(prompt: str, max_new_tokens: int = 300):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    eos_token = "<|end|>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token) if eos_token in tokenizer.get_vocab() else tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=0.7,  # Añadir algo de creatividad
            do_sample=True,
            top_p=0.9,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if "<|end|>" in output_text:
        output_text = output_text.split("<|end|>")[0] + "<|end|>"

    print("Respuesta del modelo GRPO (formato minimax):\n" + output_text)

if __name__ == "__main__":
    # Ejemplo con el nuevo formato minimax
    test_prompt = """<|board_start|>
<|0-0|><|blank|> <|0-1|><|blank|> <|0-2|><|X|>
<|1-0|><|O|> <|1-1|><|blank|> <|1-2|><|blank|>
<|2-0|><|blank|> <|2-1|><|X|> <|2-2|><|blank|>
<|board_end|>
<|turn|>bot
<|symbol|>X
<|move|>"""
    
    print("Prompt de entrada:")
    print(test_prompt)
    print("\n" + "="*50 + "\n")
    
    infer(test_prompt) 