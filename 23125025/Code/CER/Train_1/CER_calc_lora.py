import json
import Levenshtein  # pip install python-Levenshtein

# Load predictions
with open("val_predictions.json", "r", encoding="utf-8") as f:
    predictions = json.load(f)

total_chars = 0
total_edits = 0

for item in predictions:
    gt = item["ground_truth"]
    pred = item["prediction"]
    total_chars += len(gt)
    total_edits += Levenshtein.distance(gt, pred)

cer = total_edits / total_chars if total_chars > 0 else 0
print(f"Total characters: {total_chars}")
print(f"Total edits: {total_edits}")
print(f"CER: {cer:.4f}")
