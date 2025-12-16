import matplotlib.pyplot as plt

labels = [
    "<image>\\ Free OCR. (default)",
    "<image>\\ nTranscribe the Vietnamese text in the image.",
    "<image>\\ OCR Vietnamese."
]

cer_values = [0.7587, 0.6817, 0.6930]

plt.figure()
plt.bar(labels, cer_values)
plt.ylabel("CER")
plt.title("CER Comparison by Prompt")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("cer_histogram.png")
plt.close()
