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

# CLEANUP: Remove old dataset folder to prevent conflicts
if os.path.exists("dataset"):
    print("Removing old dataset folder...")
    shutil.rmtree("dataset")

os.makedirs(OUT_IMAGES, exist_ok=True)

# ---------------------------
# HELPER FUNCTION
# ---------------------------
def process_folder(src_folder, dst_folder_name):
    rows = []
    # Sort folders to ensure consistent order
    folders = sorted(os.listdir(src_folder), key=lambda x: int(x) if x.isdigit() else x)
    
    for folder in tqdm(folders, desc=f"Processing {dst_folder_name} folders"):
        folder_path = os.path.join(src_folder, folder)
        label_path = os.path.join(folder_path, "label.json")
        
        # Skip if not a directory or label file missing
        if not os.path.isdir(folder_path) or not os.path.exists(label_path):
            continue
        
        with open(label_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        
        for img_name, text in labels.items():
            src_img = os.path.join(folder_path, img_name)
            # Create destination directory inside dataset/images/dst_folder_name/folder_id
            dst_dir = os.path.join(OUT_IMAGES, dst_folder_name, folder)
            os.makedirs(dst_dir, exist_ok=True)
            dst_img = os.path.join(dst_dir, img_name)
            
            # Resize image
            if os.path.exists(src_img):
                try:
                    img = Image.open(src_img)
                    w, h = img.size
                    long_side = max(w, h)
                    scale = RESIZE_LONG_SIDE / long_side
                    new_size = (int(w*scale), int(h*scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    img.save(dst_img)
                    
                    # Normalize text
                    text = unicodedata.normalize("NFC", text)
                    # Store relative path for CSV
                    rows.append([f"{dst_folder_name}/{folder}/{img_name}", text])
                except Exception as e:
                    print(f"Error processing {src_img}: {e}")
            else:
                print(f"Missing image: {src_img}")
    
    return rows

# ---------------------------
# STEP 1: Process train_data + split train/val
# ---------------------------
# Initially save everything to "train_tmp"
all_train_rows = process_folder(TRAIN_FOLDER, "train_tmp")
random.shuffle(all_train_rows)

split_idx = int(TRAIN_RATIO * len(all_train_rows))
train_rows = all_train_rows[:split_idx]
val_rows = all_train_rows[split_idx:]

print(f"Moving {len(val_rows)} validation files...")

# Move files to proper 'val' folders
for row in val_rows:
    # row[0] is like "train_tmp/001/img.jpg"
    old_rel_path = row[0]
    new_rel_path = old_rel_path.replace("train_tmp", "val")
    
    src_path = os.path.join(OUT_IMAGES, old_rel_path)
    dst_path = os.path.join(OUT_IMAGES, new_rel_path)
    
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    # Move file if it exists
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
    
    # Update CSV record
    row[0] = new_rel_path

# Update paths in CSV records for Train rows
for row in train_rows:
    row[0] = row[0].replace("train_tmp", "train")

# FIX: Rename the directory 'train_tmp' to 'train' on the actual disk
train_tmp_dir = os.path.join(OUT_IMAGES, "train_tmp")
train_final_dir = os.path.join(OUT_IMAGES, "train")

if os.path.exists(train_tmp_dir):
    # Remove empty directories inside train_tmp that were moved to val
    # (Optional, but keeps things clean if shutil.move left empty folders)
    # Renaming the root folder
    os.rename(train_tmp_dir, train_final_dir)
else:
    print("Warning: train_tmp directory not found (maybe empty?).")

# ---------------------------
# STEP 2: Process test_data
# ---------------------------
test_rows = process_folder(TEST_FOLDER, "test")

# ---------------------------
# STEP 3: Save CSVs
# ---------------------------
print("Saving CSV files...")
os.makedirs("dataset", exist_ok=True) # Redundant but safe

def save_csv(filename, rows):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath","text"])
        writer.writerows(rows)

save_csv("dataset/train.csv", train_rows)
save_csv("dataset/val.csv", val_rows)
save_csv("dataset/test.csv", test_rows)

print("------------------------------------------------")
print(f"Processing Complete.")
print(f"Train samples: {len(train_rows)}")
print(f"Val samples:   {len(val_rows)}")
print(f"Test samples:  {len(test_rows)}")
print("Dataset ready in 'dataset/' folder.")