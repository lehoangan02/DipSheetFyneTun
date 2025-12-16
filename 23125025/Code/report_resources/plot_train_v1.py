import matplotlib.pyplot as plt

labels = ["Default prompt", "Custom prompt"]
cer = [0.1599, 0.1750]

plt.figure()
plt.bar(labels, cer)
plt.ylabel("CER")
plt.title("CER Comparison")
plt.show()
