import pandas as pd
import json
import os

# CONFIG
IMG_ROOT = "dataset/images"
CSV_FILES = {
    "train": "dataset/train.csv",
    "val": "dataset/val.csv"
}
OUTPUT_FILES = {
    "train": "dataset/deepseek_train.json",
    "val": "dataset/deepseek_val.json"
}

# The prompt the model will see
PROMPT = "<image_placeholder>Convert the text in the image to string."

def convert_csv_to_json(csv_path, output_path):
    print(f"Converting {csv_path}...")
    df = pd.read_csv(csv_path)
    data = []
    
    for _, row in df.iterrows():
        rel_path = row['filepath']
        full_path = os.path.join(IMG_ROOT, rel_path)
        
        # Verify file exists
        if not os.path.exists(full_path):
            continue
            
        entry = {
            "images": [full_path],
            "conversations": [
                {
                    "role": "User",
                    "content": PROMPT
                },
                {
                    "role": "Assistant",
                    "content": str(row['text'])
                }
            ]
        }
        data.append(entry)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} samples to {output_path}")

if __name__ == "__main__":
    convert_csv_to_json(CSV_FILES["train"], OUTPUT_FILES["train"])
    convert_csv_to_json(CSV_FILES["val"], OUTPUT_FILES["val"])