import json

with open("./datasets/tictactoe_sft_llm.jsonl", "r") as fin, open("./datasets/tictactoe_grpo_llm.jsonl", "w") as fout:
    for line in fin:
        raw = json.loads(line)
        text = raw["text"]
        split = text.split("<player_think>")
        prompt = split[0].strip()
        think_and_move = "<player_think>" + split[1].strip()
        json.dump({"prompt": prompt, "completion": think_and_move}, fout)
        fout.write("\n")
