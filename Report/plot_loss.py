import matplotlib.pyplot as plt

steps = []
losses = []

with open("train_loss.txt", "r") as f:
    for line in f:
        step, loss = line.strip().split()
        steps.append(int(step))
        losses.append(float(loss))

plt.figure()
plt.plot(steps, losses, marker='o')
plt.xlabel("Step")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()
