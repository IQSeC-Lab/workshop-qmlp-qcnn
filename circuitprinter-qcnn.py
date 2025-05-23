import pennylane as qml
from pennylane import numpy as np
from pennylane.drawer import draw_mpl
import inspect
import matplotlib.pyplot as plt

# Define device
n_qubits = 16
dev = qml.device("default.qubit", wires=n_qubits)



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


def print_weight_shapes(shapes):
    print("Parameter shapes:")
    for k, v in shapes.items():
        print(f"{k:12s}: {v}")

print_weight_shapes(weight_shapes)

dummy_input = np.zeros(16, dtype=np.float32)

# Create dummy weights to match weight_shapes
dummy_weights = []
for name in inspect.signature(qnode).parameters:
    if name == "inputs":
        continue
    shape = weight_shapes[name]
    dummy_weights.append(np.zeros(shape, dtype=np.float32))
fig, ax = draw_mpl(qnode)(dummy_input, *dummy_weights)
fig.suptitle("QCNN Conv+Pool Layer Circuit")
plt.tight_layout()
plt.show()
