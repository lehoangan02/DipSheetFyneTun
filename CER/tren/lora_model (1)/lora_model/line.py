import matplotlib.pyplot as plt

steps = [100, 600, 900, 1200, 1900, 2500, 2600]
cer = [0.4452, 0.4835, 0.4547, 0.3263, 0.2181, 0.2134, 0.2051]

plt.figure()
plt.plot(steps, cer, marker='o')
plt.xlabel("Training Step")
plt.ylabel("Character Error Rate (CER)")
plt.title("CER vs Training Step")
plt.grid(True)
plt.show()
