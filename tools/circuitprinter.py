import pennylane as qml
from pennylane import numpy as np
from pennylane.drawer import draw_mpl
import inspect
import matplotlib.pyplot as plt

# Define device
n_qubits = 16
name_fig = "qcnn.png"
dev = qml.device("default.qubit", wires=n_qubits)

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

weight_shapes = {
    "conv1": (n_qubits, 3),              # 16
    "entangle1": (n_qubits, 1),          # 16
    "pool1": (n_qubits // 2, 3),         # 8

    "conv2": (n_qubits // 2, 3),         # 8
    "entangle2": (n_qubits // 2, 1),     # 8
    "pool2": (n_qubits // 4, 3),    
}

# @qml.qnode(dev, interface="torch")
# def qnode(inputs,rot_l1,crx_l1,rot_l2 ,crx_l2):
#     # Layer 1 — Data reuploading + trainable Rot + entanglement
#     # for i in range(16):
#     #     qml.RX(inputs[i], wires=i)
#     qml.AngleEmbedding(inputs, wires=range(n_qubits))
#     for i in range(16):
#         qml.Rot(*rot_l1[i], wires=i)
#     for i in range(16):
#         qml.CRX(crx_l1[i][0], wires=[i, (i + 1) % 16])

#     # Layer 2 — Data reuploading again + Rot + CRX
#     # for i in range(16):
#     #     qml.RX(inputs[i], wires=i)
#     qml.AngleEmbedding(inputs, wires=range(n_qubits))
#     for i in range(16):
#         qml.Rot(*rot_l2[i], wires=i)
#     for i in range(16):
#         qml.CRX(crx_l2[i][0], wires=[i, (i + 1) % 16])

#     # Measurement
#     return [qml.expval(qml.PauliZ(i)) for i in range(16)]


# weight_shapes = {
#     "rot_l1": (16,3),
#     "crx_l1": (16,1),
#     "rot_l2": (16,3),
#     "crx_l2": (16,1)
# }


def print_weight_shapes(shapes):
    print("Parameter shapes:")
    for k, v in shapes.items():
        print(f"{k:12s}: {v}")

print_weight_shapes(weight_shapes)
style = "pennylane"
dummy_input = np.zeros(16, dtype=np.float32)

# Create dummy weights to match weight_shapes
dummy_weights = []
for name in inspect.signature(qnode).parameters:
    if name == "inputs":
        continue
    shape = weight_shapes[name]
    dummy_weights.append(np.zeros(shape, dtype=np.float32))
qml.drawer.use_style(style)
fig, ax = draw_mpl(qnode, show_all_wires=True)(dummy_input, *dummy_weights)
fig.set_size_inches(22, 12)  # Wide layout
# fig.suptitle("Quantum Convolutional Neural Network Circuit", fontsize=16)
plt.tight_layout()
# plt.show()
plt.savefig(name_fig)
