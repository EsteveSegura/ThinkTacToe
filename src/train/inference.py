from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "./qwen2.5-1.5b-tictactoe/checkpoint-38115"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

def infer(prompt: str, max_new_tokens: int = 32):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    eos_token = "<|end|>"
    if eos_token in tokenizer.get_vocab():
        eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    else:
        eos_token_id = tokenizer.eos_token_id  # fallback por si acaso

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

    output_tokens = outputs[0]
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)

    print("Respuesta del modelo:\n" + output_text)

if __name__ == "__main__":
    test_prompt = """<|0-0|><|X|> <|0-1|><|O|> <|0-2|><|blank|>
<|1-0|><|blank|> <|1-1|><|X|> <|1-2|><|O|>
<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>
"""
    infer(test_prompt)
