
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###############################################################################################
# Data preprocessing
###############################################################################################
bsz = 32
epochs = 20
lr = 0.001
w_decay = 1e-4

data = np.load("../dataset/AZ-Class-Task/Family_AZ_Train_Transformed.npz")

X = data["X_train"]
y_raw = data["Y_train"]  # Use multiclass malware family labels
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
# dev = qml.device("lightning.tensor", wires=n_qubits)
dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnode(inputs, conv1, entangle1, pool1, conv2, entangle2):
    # Encoding
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

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
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        out = self.qlayer(x)  # Batched input directly
        out = self.fc(out)
        return F.log_softmax(out, dim=1)







##############################################################################
# Training
##############################################################################
def train(model, DEVICE, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0.0
    total = 0.0

    for batch_idx, (inputs, target) in enumerate(train_loader, 0):
        inputs, target = inputs.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.nll_loss(outputs, target)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx+1}, Acc: {100 * correct / total:.2f}%, Loss: {running_loss / 10:.4f}")
            running_loss = 0.0

def test(model, DEVICE, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Accuracy on test set: {acc:.2f}%")
    return acc





##################################################################################
# Main loop
##################################################################################
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_acc = 0.0
model = QuantumClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
start_time = time.time()

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)

    acc = test(model, device, test_loader)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_qcnn14fam.pt")


end_time = time.time()
print(f"Training Time: {end_time - start_time:.2f} seconds")


######################################################################
# Confusion amtrix and save
#######################################################################
# Evaluation
# if os.path.exists("best_qcnn23fam.pt"):
#     print("Loading previous model...")
#     model.load_state_dict(torch.load("best_qcnn23fam.pt"))
#     model.eval()

#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
#     y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

#     with torch.no_grad():
#         output = model(X_test_tensor)
#         pred = output.argmax(dim=1)
#         true = y_test_tensor
#         acc = (pred == true).float().mean().item()
#         print(f"Test Accuracy: {acc * 100:.2f}%")

#     # Confusion matrix
#     cm = confusion_matrix(true, pred)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=[f'Pred {i}' for i in range(num_classes)],
#                 yticklabels=[f'True {i}' for i in range(num_classes)])
#     plt.title("Confusion Matrix – Quantum Classifier")
#     plt.xlabel("Predicted")
#     plt.ylabel("True")

#     plt.tight_layout()
#     plt.savefig("confusion_matrix_qcnn.png", dpi=300)
# plt.show()
