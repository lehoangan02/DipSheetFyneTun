import json
import re
from pathlib import Path

# ---------- CLEANING ----------

LATEX_RE = re.compile(r"\\\[.*?\\\]", re.DOTALL)

def clean_pred(text):
    if text is None:
        return ""

    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)

    # remove LaTeX blocks
    text = LATEX_RE.sub("", text)

    # remove leftover brackets
    text = text.replace("\\", "").strip()

    # drop obviously wrong predictions
    if len(text) > 20:
        return ""

    return text


# ---------- EDIT DISTANCE ----------

def levenshtein(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[-1][-1]


# ---------- LOAD FILES ----------

with open("label.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

preds = {}
with open("output.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "\t" not in line:
            continue
        k, v = line.split("\t", 1)
        preds[k.strip()] = v.strip()


# ---------- CER COMPUTATION ----------

total_edits = 0
total_chars = 0

for img, gt in labels.items():
    pred = preds.get(img, "")
    pred = clean_pred(pred)

    total_edits += levenshtein(gt, pred)
    total_chars += len(gt)

cer = total_edits / total_chars if total_chars > 0 else 0.0

print(f"CER = {cer:.4f}")
print(f"Total characters = {total_chars}")
print(f"Total edits = {total_edits}")
