import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import torch

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - np.max(z))  
    return e / e.sum()

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy(y_true, y_pred, eps=1e-12):
    """Binary cross-entropy."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

print("=== 1. Functions ===")
print("softmax([2.0, 1.0, 0.1]) =", np.round(softmax([2.0, 1.0, 0.1]), 3))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

def train_xor(lr=0.5, epochs=5000, seed=1):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, (2, 4)) * np.sqrt(2 / 2)  
    b1 = np.zeros((1, 4))
    W2 = rng.normal(0, 1, (4, 1)) * np.sqrt(2 / 4)
    b2 = np.zeros((1, 1))
    losses = []
    for _ in range(epochs):
        # forward pass
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        losses.append(cross_entropy(y, a2))

        n = X.shape[0]
        dz2 = (a2 - y) / n
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)          
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    preds = sigmoid(relu(X @ W1 + b1) @ W2 + b2)
    return losses, preds

print("\n=== 2. XOR training (lr=0.5, 5000 epochs) ===")
loss_main, preds = train_xor(lr=0.5)
print("Final loss:", round(loss_main[-1], 4))
print("XOR predictions after training:", np.round(preds.flatten(), 2))
print("Rounded:", (preds.flatten() > 0.5).astype(int), "| Target:", y.flatten().astype(int))

print("\n=== 3. Hand gradient vs autograd ===")
print("By hand  : dL/dw = -32, dL/db = -16")

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
x_t, t_t = torch.tensor(2.0), torch.tensor(10.0)
loss = (w * x_t + b - t_t) ** 2
loss.backward()
print(f"Autograd : w.grad = {w.grad.item()}, b.grad = {b.grad.item()}")

print("\n=== 4. Learning-rate comparison ===")
results = {}
for lr in [0.01, 0.5, 5]:
    l, p = train_xor(lr=lr)
    results[lr] = l
    print(f"lr={lr:<5} final loss={l[-1]:.4f} | preds={np.round(p.flatten(), 2)}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = {0.01: "lr=0.01 (too slow)", 0.5: "lr=0.5 (good)", 5: "lr=5 (too high)"}
for ax, (lr, l) in zip(axes, results.items()):
    ax.plot(l)
    ax.set_title(titles[lr])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("task1_loss_curves.png", dpi=150)
print("Saved plot -> task1_loss_curves.png")

print("""
=== 5. Concepts ===
Dropout:
  Randomly switches off a fraction of neurons in every training step, so the
  network cannot depend on any single neuron. This reduces overfitting; it is
  turned off at inference time (model.eval()).

BatchNorm:
  Normalises each layer's inputs over the mini-batch (zero mean, unit variance),
  then rescales with learnable gamma and beta. This stabilises and speeds up
  training and allows higher learning rates.

Early stopping:
  Monitors validation loss/accuracy and stops training when it has not improved
  for N epochs (patience). It prevents overfitting and saves the best weights.
""")
