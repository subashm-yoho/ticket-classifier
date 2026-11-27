import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)


class CsvDataset(Dataset):
    """Wraps feature matrix + labels as tensors for DataLoader."""

    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(CsvDataset(X_train_s, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(CsvDataset(X_val_s, y_val), batch_size=32, shuffle=False)

class MLP(nn.Module):
    def __init__(self, in_features=30, h1=64, h2=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),  
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, loader):
    """Validation accuracy. eval() turns off Dropout and uses BatchNorm running stats."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = (torch.sigmoid(model(xb)) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total


model = MLP()
print(f"Params: {count_params(model):,}")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 40
PATIENCE = 8  

train_losses, val_accs = [], []
best_acc, best_epoch, best_state, wait = 0.0, 0, None, 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()            
        out = model(xb)                  
        loss = criterion(out, yb)        
        loss.backward()                  
        optimizer.step()                 
        running += loss.item() * xb.size(0)

    epoch_loss = running / len(train_loader.dataset)
    val_acc = evaluate(model, val_loader)
    train_losses.append(epoch_loss)
    val_accs.append(val_acc)
    print(f"Epoch {epoch:2d} | train loss {epoch_loss:.4f} | val acc {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc, best_epoch, wait = val_acc, epoch, 0
        best_state = copy.deepcopy(model.state_dict())
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\nParams: {count_params(model):,} | Best val accuracy {best_acc:.2f} at epoch {best_epoch}")

model.load_state_dict(best_state)         
acc_original = evaluate(model, val_loader)

torch.save(best_state, "best_mlp.pt")

new_model = MLP()
new_model.load_state_dict(torch.load("best_mlp.pt"))
new_model.eval()
acc_reloaded = evaluate(new_model, val_loader)

print(f"Original (best weights) accuracy: {acc_original:.4f}")
print(f"Reloaded model accuracy: {acc_reloaded:.4f} "
      f"({'identical' if acc_original == acc_reloaded else 'DIFFERENT!'})")

logreg = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
acc_lr = accuracy_score(y_val, logreg.predict(X_val_s))
print("\n=== Comparison ===")
print(f"PyTorch MLP         : {acc_reloaded:.4f}")
print(f"Logistic Regression : {acc_lr:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(range(1, len(train_losses) + 1), train_losses)
ax[0].set_title("Training loss")
ax[0].set_xlabel("Epoch")
ax[1].plot(range(1, len(val_accs) + 1), val_accs, color="green")
ax[1].axvline(best_epoch, color="red", linestyle="--", label=f"best epoch {best_epoch}")
ax[1].set_title("Validation accuracy")
ax[1].set_xlabel("Epoch")
ax[1].legend()
plt.tight_layout()
plt.savefig("task2_training_curves.png", dpi=150)
print("Saved plot -> task2_training_curves.png")
