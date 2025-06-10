
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  only if running it on the server
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

bsz = 64
epochs = 20
lr = 0.001
w_decay = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################################################################################
# Data preprocessing
###############################################################################################
best_model = "qcnn-ember23-run3-mod.pt"
cm_name= "cm_qcnn-ember23-run3-mod.png"
data_train = np.load("dataset/Ember-Class-100class/Ember-Class-100class_23_families_train.npz") 
data_test = np.load("dataset/Ember-Class-100class/Ember-Class-100class_23_families_test.npz") 



X_train = data_train["X_train"]
y_train_raw = data_train["Y_train"]
# y_train = np.argmax(y_train_raw,axis=1)
y_train = y_train_raw
X_test = data_test["X_test"]
y_test_raw = data_test["Y_test"]
# y_test = np.argmax(y_test_raw, axis=1)
y_test = y_test_raw

# num_classes = y_train_raw.shape[1]
num_classes = len(np.unique(y_train_raw))
# print(num_classes)

if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()
if hasattr(X_test, "toarray"):
    X_test = X_test.toarray()

# Normalize to [0, 1]
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=16)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Convert to tensors
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)
# DataLoaders
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=bsz)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=bsz)

###############################################################################################
n_qubits = 16


##### Sim #####
dev = qml.device("default.qubit", wires=n_qubits)
# dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnode(inputs, conv1, entangle1, pool1, conv2, entangle2, pool2):
    # Encoding
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    # --- Layer 1: conv + CRX entanglement ---
    for i in range(n_qubits):
        qml.Rot(*conv1[i], wires=i)
    for i in range(n_qubits):
        qml.CRX(entangle1[i][0], wires=[i, (i + 1) % n_qubits])

    # --- Pooling: keep even qubits ---
    qubits_l1 = list(range(0, n_qubits, 2))  # 0, 2, ..., 14
    for i, w in enumerate(qubits_l1):
        qml.Rot(*pool1[i], wires=w)

    # --- Layer 2: conv + CRX entanglement on 8 qubits ---
    for i, w in enumerate(qubits_l1):
        qml.Rot(*conv2[i], wires=w)
    for i in range(len(qubits_l1)):
        qml.CRX(entangle2[i][0], wires=[qubits_l1[i], qubits_l1[(i + 1) % len(qubits_l1)]])
    qubits_l2 = list(range(0, len(qubits_l1), 2)) 
    for i, w in enumerate([qubits_l1[j] for j in qubits_l2]):
        qml.Rot(*pool2[i], wires=w)
    final_qubits = [qubits_l1[j] for j in qubits_l2]
    return [qml.expval(qml.PauliZ(q)) for q in final_qubits]


n_qubits = 16

weight_shapes = {
    "conv1": (n_qubits, 3),              # 16
    "entangle1": (n_qubits, 1),          # 16
    "pool1": (n_qubits // 2, 3),         # 8

    "conv2": (n_qubits // 2, 3),         # 8
    "entangle2": (n_qubits // 2, 1),     # 8
    "pool2": (n_qubits // 4, 3),         # 4

    # "conv3": (n_qubits // 4, 3),         # 4
    # "entangle3": (n_qubits // 4, 1),     # 4
    # "pool3": (n_qubits // 8, 3),         # 2

    # "conv4": (n_qubits // 8, 3),         # 2
    # "entangle4": (n_qubits // 8, 1),     # 2
    # "pool4": (n_qubits // 16, 3),        # 1

    # "conv5": (n_qubits // 16, 3),        # 1
    # "entangle5": (n_qubits // 16, 1),    # 1
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
        torch.save(model.state_dict(), best_model)
end_time = time.time()
print(f"Training Time: {end_time - start_time:.2f} seconds")


######################################################################
# Confusion amtrix and save
#######################################################################
# Evaluation
if os.path.exists(best_model):
    print("Loading previous model...")
    model.load_state_dict(torch.load(best_model))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(X_test_tensor)
        pred = output.argmax(dim=1)
        true = y_test_tensor
        acc = (pred == true).float().mean().item()
        print(f"Test Accuracy: {acc * 100:.2f}%")

    # Confusion matrix
    # cm = confusion_matrix(true, pred)
    cm = confusion_matrix(true.cpu().numpy(), pred.cpu().numpy())

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Pred {i}' for i in range(num_classes)],
                yticklabels=[f'True {i}' for i in range(num_classes)])
    plt.title("Confusion Matrix – Quantum Classifier")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig(cm_name, dpi=300)
else:
    print(f'{best_model} not found')