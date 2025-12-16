import json
from difflib import SequenceMatcher

# Load files
with open("val_predictions.json", "r", encoding="utf-8") as f:
    val_preds = json.load(f)

with open("labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

# Function to compute similarity (1 = exact match, 0 = totally different)
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

results = []
for item in val_preds:
    filename = item["filename"]
    gt = labels.get(filename, "")
    pred = item["prediction"]
    sim = similarity(gt.lower(), pred.lower())
    results.append({
        "filename": filename,
        "ground_truth": gt,
        "prediction": pred,
        "similarity": sim
    })

# Sort by similarity
results_sorted = sorted(results, key=lambda x: x["similarity"])

top_20_worst = results_sorted[:20]
top_20_best = results_sorted[-20:]

# Print
print("Top 20 Worst Predictions:")
for r in top_20_worst:
    print(r)

print("\nTop 20 Best Predictions:")
for r in top_20_best:
    print(r)
