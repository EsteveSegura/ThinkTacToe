from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "./qwen2.5-0.5b-tictactoe-sft-nothink-minmax/checkpoint-156"

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
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    if "<|end|>" in output_text:
        output_text = output_text.split("<|end|>")[0] + "<|end|>"

    print("Respuesta del modelo:\n" + output_text)

tok_ids = tokenizer.encode("<|0-0|>", add_special_tokens=False)
print(tok_ids)                             # ¿sólo un id?
print(tokenizer.decode(tok_ids,            # → ''
                      skip_special_tokens=True))
print(tokenizer.decode(tok_ids,            # → '<|0-0|>'
                      skip_special_tokens=False))

if __name__ == "__main__":
    test_prompt = """<|board_start|>
<|0-0|><|blank|> <|0-1|><|blank|> <|0-2|><|O|>
<|1-0|><|blank|> <|1-1|><|blank|> <|1-2|><|blank|>
<|2-0|><|X|> <|2-1|><|blank|> <|2-2|><|blank|>
<|board_end|>
<|turn|>bot
<|symbol|>X
<|move|>
"""
    infer(test_prompt)


