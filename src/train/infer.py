from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Ruta del modelo fine-tuneado
model_name = "./Qwen2.5-1.5B-DPO/checkpoint-6399"

# Detectar dispositivo disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo y tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

def infer(prompt: str, max_new_tokens: int = 300) -> str:
    # Tokenizar prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(device)

    # Detectar token de fin
    eos_token = "<|end|>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    # Generar predicción
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

    # Extraer solo los nuevos tokens generados
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Recortar en caso de que se incluya el token <|end|>
    if eos_token in output_text:
        output_text = output_text.split(eos_token)[0] + eos_token

    return output_text

if __name__ == "__main__":
    # Prompt de prueba
    test_prompt = """<|0-0|><|X|> <|0-1|><|O|> <|0-2|><|blank|>
<|1-0|><|blank|> <|1-1|><|X|> <|1-2|><|O|>
<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>
"""

    result = infer(test_prompt)
    print("Respuesta del modelo:\n" + result)
