from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Ruta del checkpoint fine-tuneado
model_path = "./Qwen2.5-1.5B-DPO/checkpoint-6399"

# Inicializar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo y tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()

# Establecer pad token si falta
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Función de inferencia
def infer(prompt: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(device)

    eos_token = "<|end|>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Recortar si incluye <|end|>
    if eos_token in decoded:
        decoded = decoded.split(eos_token)[0] + eos_token

    return decoded.strip()

# Ejemplo de prompt realista (mismo formato que entrenamiento)
if __name__ == "__main__":
    prompt = """<|0-0|><|X|> <|0-1|><|O|> <|0-2|><|blank|>
<|1-0|><|blank|> <|1-1|><|X|> <|1-2|><|O|>
<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>

think:"""

    output = infer(prompt)
    print("\n=== RESPUESTA DEL MODELO ===\n")
    print(output)
