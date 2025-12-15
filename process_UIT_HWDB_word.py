import os
import json
import shutil
import random

random.seed(42)

BASE_DIR = "UIT_HWDB_word"
OUTPUT_DIR = "UIT_HWDB_word_clean"

TRAIN_RATIO = 0.8

def prepare_output():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)

def process_split(input_dir, split_name, do_split=False):
    all_samples = []

    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        label_path = os.path.join(folder_path, "label.json")
        if not os.path.exists(label_path):
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        for img_name, text in labels.items():
            img_path = os.path.join(folder_path, img_name)
            if not os.path.exists(img_path):
                continue

            new_name = f"{folder}_{img_name}"
            all_samples.append((img_path, new_name, text))

    if do_split:
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * TRAIN_RATIO)
        return {
            "train": all_samples[:split_idx],
            "val": all_samples[split_idx:]
        }
    else:
        return {split_name: all_samples}

def write_split(samples, split_name):
    label_dict = {}
    img_out_dir = os.path.join(OUTPUT_DIR, split_name, "images")

    for src, new_name, text in samples:
        dst = os.path.join(img_out_dir, new_name)
        shutil.copy(src, dst)
        label_dict[new_name] = text

    with open(os.path.join(OUTPUT_DIR, split_name, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(label_dict, f, ensure_ascii=False, indent=2)

def main():
    prepare_output()

    # Train → train + val
    train_samples = process_split(
        os.path.join(BASE_DIR, "train_data"),
        "train",
        do_split=True
    )

    write_split(train_samples["train"], "train")
    write_split(train_samples["val"], "val")

    # Test (no split)
    test_samples = process_split(
        os.path.join(BASE_DIR, "test_data"),
        "test",
        do_split=False
    )

    write_split(test_samples["test"], "test")

    print("✅ Dataset successfully merged and split!")

if __name__ == "__main__":
    main()
