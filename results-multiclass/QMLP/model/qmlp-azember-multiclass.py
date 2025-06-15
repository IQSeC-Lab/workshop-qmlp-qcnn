import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # setting
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
import matplotlib.pyplot as plt
import math

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_recall_curve, auc
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix
from pennylane.qnn import TorchLayer
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import qiskit_aer.noise as noise
from qiskit_aer.noise import NoiseModel


bsz = 64
epochs = 20
lr = 0.001
w_decay = 1e-4



best_model = "qmlp-API23-run3-mod.pt"
cm_name= "cm_qmlp-API23-run3-mod.png"
filename = "qmlp-API23-run3.txt"
############################################################################################################################################################################

data_train = np.load("../dataset/API_graph/APIgraph_train23-fam.npz") 
data_test = np.load("../dataset/API_graph/APIgraph_test23-fam.npz") 

# API Graph
X_train = data_train["X"]
y_train_raw = data_train["y_multilabel"]
X_test = data_test["X"]
y_test_raw = data_test["y_multilabel"]
y_train = np.argmax(y_train_raw, axis=1)
y_test = np.argmax(y_test_raw, axis=1)

# Ember and AZ
# X_train = data_train["X_train"]
# y_train_raw = data_train["Y_train"]
# y_train = y_train_raw
# X_test = data_test["X_test"]
# y_test_raw = data_test["Y_test"]
# y_test = y_test_raw

num_classes = len(np.unique(y_train))  

# num_classes = len(np.unique(y_train_raw))

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

# print("going now to the quantum model")
###########################################################################################################################################################################
noise_model = NoiseModel(basis_gates=['id', 'rz', 'sx', 'cx', 'x'])

# phase flip
p = 0.01
phaseflip = noise.pauli_error([('Z', p), ('I', 1-p)])
phaseflip2 = noise.pauli_error([('Z', p), ('I', 1-p)])
noisemy = phaseflip.tensor(phaseflip2)
noise_model.add_all_qubit_quantum_error(phaseflip, ['id', 'rz', 'sx', 'x'])
noise_model.add_all_qubit_quantum_error(noisemy, ['cx'])

###########################################################################################################################################################################
n_qubits = 16

# HERE CHANGE THE SIMULATOR

dev = qml.device("default.qubit", wires=n_qubits)      # w/o noise


@qml.qnode(dev, interface="torch")
def qnode(inputs,rot_l1,crx_l1,rot_l2 ,crx_l2):
    # Layer 1 — Data reuploading + trainable Rot + entanglement

    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    for i in range(16):
        qml.Rot(*rot_l1[i], wires=i)
    for i in range(16):
        qml.CRX(crx_l1[i][0], wires=[i, (i + 1) % 16])

    # Layer 2 — Data reuploading again + Rot + CRX
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    for i in range(16):
        qml.Rot(*rot_l2[i], wires=i)
    for i in range(16):
        qml.CRX(crx_l2[i][0], wires=[i, (i + 1) % 16])

    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(16)]


weight_shapes = {
    "rot_l1": (16,3),
    "crx_l1": (16,1),
    "rot_l2": (16,3),
    "crx_l2": (16,1)

}

#################################################################################################################################################################################
qlayer = TorchLayer(qnode, weight_shapes)
class drebin(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qlayer
        self.fc = nn.Linear(16, num_classes)
    def forward(self,x):
        x = x.to(next(self.parameters()).device)
        out = self.qlayer(x)  # Batched input directly
        out = self.fc(out)
        return F.log_softmax(out, dim=1)




###########################################################################################################################################################################
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

###########################################################################################################################################################################
best_acc = 0.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = drebin().to(device)

################################################################################################
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)

start_time = time.time()

for epoch in range(epochs):
    train(model, device, train_loader, optimizer, epoch)

    acc = test(model, device, test_loader)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), best_model)


end_time = time.time()
print(f"Training Time: {end_time - start_time:.2f} seconds")
########################################################################




######################################################################
# Confusion amtrix and save
#######################################################################
# Evaluation
if os.path.exists(best_model):
    print("Loading previous model...")
    model.load_state_dict(torch.load(best_model))
    model.eval()
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    outputs = []
    y_true = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            out = model(x_batch)
            outputs.append(out)
            y_true.append(y_batch)

    output = torch.cat(outputs, dim=0)
    true = torch.cat(y_true, dim=0)

    num_classes = output.shape[1]
    min_label = true.min().item()
    max_label = true.max().item()
    print(f"Label range: {min_label} to {max_label}, num_classes: {num_classes}")

    
    pred = output.argmax(dim=1)
    acc = (pred == true).float().mean().item()

    # Confusion matrix
    y_true = true.cpu().numpy()
    y_pred = pred.cpu().numpy()
    probs = torch.softmax(output, dim=1).cpu().numpy()
    cm = confusion_matrix(true.cpu().numpy(), pred.cpu().numpy())

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    precision_macro = np.mean(TP / (TP + FP + 1e-8))
    recall_macro = np.mean(TP / (TP + FN + 1e-8))
    f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro + 1e-8)
    fpr_macro = np.mean(FP / (FP + TN + 1e-8))
    fnr_macro = np.mean(FN / (FN + TP + 1e-8))
    test_loss = F.nll_loss(output, true).item()


    if num_classes == 2:
        roc_auc = roc_auc_score(y_true, probs[:, 1])
        pr_curve, rc_curve, _ = precision_recall_curve(y_true, probs[:, 1])
        pr_auc = auc(rc_curve, pr_curve)
    else:
        roc_auc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
        pr_auc = float('nan')
    #     # ------------------------- Print Results ---------------------------

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy:         {acc:.4f}")
    print(f"Loss:             {test_loss:.4f}")
    print(f"Precision (macro):{precision_macro:.4f}")
    print(f"Recall (macro):   {recall_macro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"FPR (macro):      {fpr_macro:.4f}")
    print(f"FNR (macro):      {fnr_macro:.4f}")
    print(f"ROC-AUC (macro):  {roc_auc:.4f}")
    print(f"PR-AUC (macro):   {pr_auc:.4f}")

    with open(filename, 'w') as f:
        f.write("=== Evaluation Metrics ===\n")
        f.write(f"Accuracy:         {acc:.4f}\n")
        f.write(f"Loss:             {test_loss:.4f}\n")
        f.write(f"Precision (macro):{precision_macro:.4f}\n")
        f.write(f"Recall (macro):   {recall_macro:.4f}\n")
        f.write(f"F1 Score (macro): {f1_macro:.4f}\n")
        f.write(f"FPR (macro):      {fpr_macro:.4f}\n")
        f.write(f"FNR (macro):      {fnr_macro:.4f}\n")
        f.write(f"ROC-AUC (macro):  {roc_auc:.4f}\n")
        f.write(f"PR-AUC (macro):   {pr_auc:.4f}\n")


    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Pred {i}' for i in range(num_classes)],
                yticklabels=[f'True {i}' for i in range(num_classes)])
    plt.title("Confusion Matrix – Quantum Classifier")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    # plt.savefig(cm_name, dpi=300)

else: 
    print(f'{best_model} not found')

