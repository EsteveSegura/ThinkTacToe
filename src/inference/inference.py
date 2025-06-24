from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "./qwen2.5-0.5b-tictactoe-sft-nothink-minmax/checkpoint-78"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

def infer(prompt: str, max_new_tokens: int = 600):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    eos_token = "<|end|>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token) if eos_token in tokenizer.get_vocab() else tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            do_sample=False,  # Usar greedy decoding para consistencia
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Limpiar la salida para mostrar solo el movimiento
    if "<|end|>" in output_text:
        output_text = output_text.split("<|end|>")[0]
    
    # Remover tokens especiales del movimiento
    move_text = output_text.replace("<|0-0|>", "0-0").replace("<|0-1|>", "0-1").replace("<|0-2|>", "0-2")
    move_text = move_text.replace("<|1-0|>", "1-0").replace("<|1-1|>", "1-1").replace("<|1-2|>", "1-2")
    move_text = move_text.replace("<|2-0|>", "2-0").replace("<|2-1|>", "2-1").replace("<|2-2|>", "2-2")
    
    print("Prompt completo:")
    print(prompt)
    print("\nRespuesta del modelo (raw):")
    print(output_text)
    print("\nMovimiento extraído:")
    print(move_text.strip())

if __name__ == "__main__":
    # Prompt que coincide exactamente con el formato del dataset
    test_prompt = """<|board_start|>
<|0-0|><|blank|> <|0-1|><|blank|> <|0-2|><|O|>
<|1-0|><|blank|> <|1-1|><|blank|> <|1-2|><|blank|>
<|2-0|><|X|> <|2-1|><|blank|> <|2-2|><|blank|>
<|board_end|>
<|turn|>bot
<|symbol|>X
<|move|>"""
    
    infer(test_prompt)
    
    # Probar con otro prompt del dataset
    print("\n" + "="*50)
    print("SEGUNDA PRUEBA - Tablero vacío:")
    print("="*50)
    
    test_prompt2 = """<|board_start|>
<|0-0|><|blank|> <|0-1|><|blank|> <|0-2|><|blank|>
<|1-0|><|blank|> <|1-1|><|blank|> <|1-2|><|blank|>
<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>
<|board_end|>
<|turn|>bot
<|symbol|>X
<|move|>"""
    
    infer(test_prompt2)


