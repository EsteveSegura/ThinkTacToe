from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "./qwen1.5b-tictactoe"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval()

def infer(prompt: str, max_new_tokens: int = 32):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    test_prompt = """<|0-0|><|X|> <|0-1|><|O|> <|0-2|><|blank|>
<|1-0|><|blank|> <|1-1|><|X|> <|1-2|><|O|>
<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>
<think>The opponent is about to win via the top row, I block at 0-2.</think>
"""
    infer(test_prompt)
