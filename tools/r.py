import os, json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_recall_curve, auc
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from pennylane import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pennylane.qnn import TorchLayer
import pennylane as qml
import torch.nn as nn

# ---------------------------- Model Definition ----------------------------

n_qubits = 16
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnode(inputs,rot_l1,crx_l1,rot_l2 ,crx_l2):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    for i in range(16):
        qml.Rot(*rot_l1[i], wires=i)
    for i in range(16):
        qml.CRX(crx_l1[i][0], wires=[i, (i + 1) % 16])
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    for i in range(16):
        qml.Rot(*rot_l2[i], wires=i)
    for i in range(16):
        qml.CRX(crx_l2[i][0], wires=[i, (i + 1) % 16])
    return [qml.expval(qml.PauliZ(i)) for i in range(16)]

weight_shapes = {
    "rot_l1": (16,3), "crx_l1": (16,1),
    "rot_l2": (16,3), "crx_l2": (16,1)
}

qlayer = TorchLayer(qnode, weight_shapes)

class drebin(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.qlayer = qlayer
        self.fc = nn.Linear(16, num_classes)
    def forward(self,x):
        x = x.to(next(self.parameters()).device)
        out = self.qlayer(x)
        return F.log_softmax(self.fc(out), dim=1)

# -------------------------- Data Preprocessing ----------------------------

# Load test data only
best_model = "../results/QMLP/z-saved-models/Ember/qmlp-ember4-run1-mod.pt"
# cm_name = "cm_qmlp-ember4-run3-mod.png"
metricseval= "qmlp-ember4-run1.json"
data_test = np.load("../dataset/Ember-Class-100class/Ember-Class-100class_4_families_test.npz")
X_test = data_test["X_test"]
y_test = data_test["Y_test"]

num_classes = len(np.unique(y_test))
if hasattr(X_test, "toarray"):
    X_test = X_test.toarray()

scaler = MinMaxScaler()
X_test = scaler.fit_transform(X_test)

pca = PCA(n_components=16)
X_test = pca.fit_transform(X_test)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# ------------------------- Load & Evaluate Model ---------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = drebin(num_classes).to(device)


if os.path.exists(best_model):
    print("Loading model...")
    model.load_state_dict(torch.load(best_model))
    model.eval()

    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    with torch.no_grad():
        output = model(X_test_tensor)
        pred = output.argmax(dim=1)
        true = y_test_tensor
        acc = (pred == true).float().mean().item()

    # Metrics
    y_true = true.cpu().numpy()
    y_pred = pred.cpu().numpy()
    probs = torch.softmax(output, dim=1).cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    precision_macro = np.mean(TP / (TP + FP + 1e-8))
    recall_macro = np.mean(TP / (TP + FN + 1e-8))
    f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro + 1e-8)
    fpr_macro = np.mean(FP / (FP + TN + 1e-8))
    fnr_macro = np.mean(FN / (FN + TP + 1e-8))
    test_loss = F.nll_loss(output, y_test_tensor).item()

    if num_classes == 2:
        roc_auc = roc_auc_score(y_true, probs[:, 1])
        pr_curve, rc_curve, _ = precision_recall_curve(y_true, probs[:, 1])
        pr_auc = auc(rc_curve, pr_curve)
    else:
        roc_auc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
        pr_auc = float('nan')

    metrics = {
        "accuracy": acc,
        "loss": test_loss,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "fpr_macro": fpr_macro,
        "fnr_macro": fnr_macro,
        "roc_auc_macro": roc_auc,
        "pr_auc_macro": pr_auc
    }

    with open(metricseval, "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Evaluation complete. Metrics saved.")

    # Plot CM
    import matplotlib.pyplot as plt
    import seaborn as sns
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
    print(f"❌ {best_model} not found.")
