
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

import pennylane as qml
from pennylane.qnn import TorchLayer

#################################################################################################
# Set random seed for reproducibility
#################################################################################################
torch.manual_seed(42)
np.random.seed(42)

###############################################################################################
# Data preprocessing
###############################################################################################
bsz = 64
epochs = 20
lr = 0.001
w_decay = 1e-4

data = np.load("dataset/2012_2013_filtered.npz")

X = data["X"]
y_raw = data["y_multilabel"]  # Use multiclass malware family labels
y = np.argmax(y_raw, axis=1)
num_classes = y_raw.shape[1]


if hasattr(X, "toarray"):
    X = X.toarray()

X = MinMaxScaler().fit_transform(X)
X = PCA(n_components=16).fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=bsz)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=bsz)

###############################################################################################
n_qubits = 16


##### Sim #####
device = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnode(inputs, conv1, entangle1, pool1, conv2, entangle2):
    # Encoding
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)

    # --- Layer 1: conv + CRX entanglement ---
    for i in range(n_qubits):
        qml.Rot(*conv1[i], wires=i)
    for i in range(n_qubits):
        qml.CRX(entangle1[i][0], wires=[i, (i + 1) % n_qubits])

    # --- Pooling: keep even qubits ---
    kept = list(range(0, n_qubits, 2))  # 0, 2, ..., 14
    for i, w in enumerate(kept):
        qml.Rot(*pool1[i], wires=w)

    # --- Layer 2: conv + CRX entanglement on 8 qubits ---
    for i, w in enumerate(kept):
        qml.Rot(*conv2[i], wires=w)
    for i in range(len(kept)):
        qml.CRX(entangle2[i][0], wires=[kept[i], kept[(i + 1) % len(kept)]])

    # Output from 1 qubit (e.g., qubit 0)
    return [qml.expval(qml.PauliZ(q)) for q in kept[:4]]


weight_shapes = {
    "conv1": (16, 3),        
    "entangle1": (16, 1),   
    "pool1": (8, 3),         
    "conv2": (8, 3),         
    "entangle2": (8, 1),    
}


qlayer = TorchLayer(qnode, weight_shapes)

# PyTorch model
class QuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qlayer
        self.fc = nn.Linear(4, 10)

    def forward(self, x):
        x = self.qlayer(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

model = QuantumClassifier()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
criterion = nn.NLLLoss()



##############################################################################
# Training
##############################################################################
best_loss = float('inf')
patience, trials = 5, 0
start_time = time.time()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        trials = 0
        best_model = model.state_dict()
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping")
            break

end_time = time.time()
print(f"Training Time: {end_time - start_time:.2f} seconds")

# Evaluation
model.load_state_dict(best_model)
model.eval()

with torch.no_grad():
    output = model(X_test_tensor)
    pred = output.argmax(dim=1)
    true = y_test_tensor
    acc = (pred == true).float().mean().item()
    print(f"Test Accuracy: {acc * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(true, pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Pred {i}' for i in range(num_classes)],
            yticklabels=[f'True {i}' for i in range(num_classes)])
plt.title("Confusion Matrix – Quantum Classifier")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.savefig("confusion_matrix_qcnn.png", dpi=300)
# plt.show()
