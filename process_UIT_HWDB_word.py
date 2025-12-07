import os
import json
import csv
import unicodedata
from tqdm import tqdm
from PIL import Image
import shutil
import random

# ---------------------------
# CONFIG
# ---------------------------
ROOT = "DATA/UIT_HWDB_word"
TRAIN_FOLDER = os.path.join(ROOT, "train_data")
TEST_FOLDER = os.path.join(ROOT, "test_data")

OUT_IMAGES = "dataset/images"
RESIZE_LONG_SIDE = 1024
TRAIN_RATIO = 0.9  # 90% train, 10% val

os.makedirs(OUT_IMAGES, exist_ok=True)

# ---------------------------
# HELPER FUNCTION
# ---------------------------
def process_folder(src_folder, dst_folder_name):
    rows = []
    folders = sorted(os.listdir(src_folder), key=lambda x: int(x))
    
    for folder in tqdm(folders, desc=f"Processing {dst_folder_name} folders"):
        folder_path = os.path.join(src_folder, folder)
        label_path = os.path.join(folder_path, "label.json")
        if not os.path.isdir(folder_path) or not os.path.exists(label_path):
            continue
        
        with open(label_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        
        for img_name, text in labels.items():
            src_img = os.path.join(folder_path, img_name)
            dst_dir = os.path.join(OUT_IMAGES, dst_folder_name, folder)
            os.makedirs(dst_dir, exist_ok=True)
            dst_img = os.path.join(dst_dir, img_name)
            
            # Resize image
            if os.path.exists(src_img):
                img = Image.open(src_img)
                w, h = img.size
                long_side = max(w, h)
                scale = RESIZE_LONG_SIDE / long_side
                new_size = (int(w*scale), int(h*scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                img.save(dst_img)
            
            # Normalize text
            text = unicodedata.normalize("NFC", text)
            rows.append([f"{dst_folder_name}/{folder}/{img_name}", text])
    
    return rows

# ---------------------------
# STEP 1: Process train_data + split train/val
# ---------------------------
all_train_rows = process_folder(TRAIN_FOLDER, "train_tmp")
random.shuffle(all_train_rows)

split_idx = int(TRAIN_RATIO * len(all_train_rows))
train_rows = all_train_rows[:split_idx]
val_rows = all_train_rows[split_idx:]

# Move files to proper folders
for row in val_rows:
    src_path = os.path.join(OUT_IMAGES, row[0])
    dst_path = os.path.join(OUT_IMAGES, "val", "/".join(row[0].split("/")[1:]))
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.move(src_path, dst_path)
    row[0] = row[0].replace("train_tmp", "val")

for row in train_rows:
    row[0] = row[0].replace("train_tmp", "train")

# ---------------------------
# STEP 2: Process test_data
# ---------------------------
test_rows = process_folder(TEST_FOLDER, "test")

# ---------------------------
# STEP 3: Save CSVs
# ---------------------------
os.makedirs("dataset", exist_ok=True)

with open("dataset/train.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath","text"])
    writer.writerows(train_rows)

with open("dataset/val.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath","text"])
    writer.writerows(val_rows)

with open("dataset/test.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath","text"])
    writer.writerows(test_rows)

print(f"Done! Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")
print("Dataset ready in dataset/ folder.")
