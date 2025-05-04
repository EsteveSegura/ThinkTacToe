
import json

def prepare_dataset(input_path: str, output_path: str):
    with open(input_path, "r") as infile:
        raw_data = json.load(infile)

    with open(output_path, "w") as outfile:
        for sample in raw_data:
            text = f"{sample['board']}\n{sample['think']}\n{sample['move']}"
            json.dump({"text": text}, outfile)
            outfile.write("\n")

if __name__ == "__main__":
    prepare_dataset("tictactoe_raw.json", "tictactoe_hf.json")
