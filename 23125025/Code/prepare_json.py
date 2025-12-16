import pandas as pd
import os
import json

# ---------------------------
# CONFIG
# ---------------------------
IMG_ROOT = "dataset/images"
CSV_FILES = {
    "train": "dataset/train.csv",
    "val": "dataset/val.csv",
    "test": "dataset/test.csv"  # optional
}
OUTPUT_FILES = {
    "train": "dataset/deepseek_train.json",
    "val": "dataset/deepseek_val.json",
    "test": "dataset/deepseek_test.json"  # optional
}

# The prompt the model sees before each image
PROMPT = "<image>\nConvert the text in the image to string."

ROLE_MAP = {
    "User": "<|User|>",
    "Assistant": "<|Assistant|>"
}

# ---------------------------
# CONVERSION FUNCTION
# ---------------------------
def convert_csv_to_unsloth_json(csv_path, output_path):
    print(f"Converting {csv_path} -> {output_path}")
    df = pd.read_csv(csv_path)
    data = []

    for _, row in df.iterrows():
        rel_path = row['filepath']
        full_path = os.path.join(IMG_ROOT, rel_path)

        # Skip if image does not exist
        if not os.path.exists(full_path):
            print(f"Missing image: {full_path}")
            continue

        entry = {
            "images": [full_path],
            "messages": [   # Note: Unsloth expects 'messages'
                {
                    "role": ROLE_MAP["User"],
                    "content": PROMPT,
                    "images": [full_path]  # Include image path here
                },
                {
                    "role": ROLE_MAP["Assistant"],
                    "content": str(row['text'])
                }
            ]
        }
        data.append(entry)

    # Save JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} samples to {output_path}")

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    for split in ["train", "val"]:
        convert_csv_to_unsloth_json(CSV_FILES[split], OUTPUT_FILES[split])

    # Optional: convert test set for evaluation
    if os.path.exists(CSV_FILES["test"]):
        convert_csv_to_unsloth_json(CSV_FILES["test"], OUTPUT_FILES["test"])
