import json
import random

with open("tictac_dpo_dataset.json") as f:
    raw_data = json.load(f)

dpo_dataset = []

for item in raw_data:
    prompt = item["board"]
    oks = item["ok"]
    kos = item["ko"]

    # emparejamos cada buena con una mala aleatoria
    for ok in oks:
        if not kos:
            continue
        ko = random.choice(kos)
        dpo_dataset.append({
            "prompt": prompt,
            "chosen": ok["think"] + "\n" + ok["move"],
            "rejected": ko["think"] + "\n" + ko["move"]
        })

with open("tictactoe_dpo.json", "w") as f:
    json.dump(dpo_dataset, f, indent=2)

