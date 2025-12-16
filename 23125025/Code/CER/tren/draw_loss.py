import matplotlib.pyplot as plt

# Replace 'loss.txt' with your filename
filename = 'loss.txt'

steps = []
loss = []

with open(filename, 'r') as f:
    for line in f:
        if line.strip() == '':
            continue
        step, value = line.strip().split()
        steps.append(int(step))
        loss.append(float(value))

plt.figure(figsize=(10,6))
plt.plot(steps, loss, marker='.', linestyle='-', color='blue', markersize=3)
plt.title("Loss over Steps")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
