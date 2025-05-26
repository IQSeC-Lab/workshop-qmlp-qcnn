import pennylane as qml
from pennylane import numpy as np
from pennylane.drawer import draw_mpl
import inspect
import matplotlib.pyplot as plt

# Define device
n_qubits = 16
dev = qml.device("default.qubit", wires=n_qubits)



# @qml.qnode(dev, interface="torch")
# def qnode(inputs, conv1, entangle1, pool1, conv2, entangle2):
#     # Encoding
#     qml.AngleEmbedding(inputs, wires=range(n_qubits))

#     # --- Layer 1: conv + CRX entanglement ---
#     for i in range(n_qubits):
#         qml.Rot(*conv1[i], wires=i)
#     for i in range(n_qubits):
#         qml.CRX(entangle1[i][0], wires=[i, (i + 1) % n_qubits])

#     # --- Pooling: keep even qubits ---
#     kept = list(range(0, n_qubits, 2))  # 0, 2, ..., 14
#     for i, w in enumerate(kept):
#         qml.Rot(*pool1[i], wires=w)

#     # --- Layer 2: conv + CRX entanglement on 8 qubits ---
#     for i, w in enumerate(kept):
#         qml.Rot(*conv2[i], wires=w)
#     for i in range(len(kept)):
#         qml.CRX(entangle2[i][0], wires=[kept[i], kept[(i + 1) % len(kept)]])

#     # Output from 1 qubit (e.g., qubit 0)
#     return [qml.expval(qml.PauliZ(q)) for q in kept[:4]]


# weight_shapes = {
#     "conv1": (16, 3),        
#     "entangle1": (16, 1),   
#     "pool1": (8, 3),         
#     "conv2": (8, 3),         
#     "entangle2": (8, 1),    
# }



@qml.qnode(dev)
def qnode(inputs, w000, w001, w002, w003, w004, w005, w006, w007, w008, w009, w010, w011, w012, w013, w014, w015,
          w100, w101, w102, w103, w104, w105, w106, w107, w108, w109, w110, w111, w112, w113, w114, w115,
          x000, x001, x002, x003, x004, x005, x006, x007, x008, x009, x010, x011, x012, x013, x014, x015,
          x100, x101, x102, x103, x104, x105, x106, x107, x108, x109, x110, x111, x112, x113, x114, x115,):
    for rx in range(16):
      qml.RX(inputs[rx], wires=rx)
    qml.Rot(*w000, wires=0)
    qml.Rot(*w001, wires=1)
    qml.Rot(*w002, wires=2)
    qml.Rot(*w003, wires=3)
    qml.Rot(*w004, wires=4)
    qml.Rot(*w005, wires=5)
    qml.Rot(*w006, wires=6)
    qml.Rot(*w007, wires=7)
    qml.Rot(*w008, wires=8)
    qml.Rot(*w009, wires=9)
    qml.Rot(*w010, wires=10)
    qml.Rot(*w011, wires=11)
    qml.Rot(*w012, wires=12)
    qml.Rot(*w013, wires=13)
    qml.Rot(*w014, wires=14)
    qml.Rot(*w015, wires=15)
    
    qml.CRX(x000, wires=[0,1])
    qml.CRX(x001, wires=[1,2])
    qml.CRX(x002, wires=[2,3])
    qml.CRX(x003, wires=[3,4])
    qml.CRX(x004, wires=[4,5])
    qml.CRX(x005, wires=[5,6])
    qml.CRX(x006, wires=[6,7])
    qml.CRX(x007, wires=[7,8])
    qml.CRX(x008, wires=[8,9])
    qml.CRX(x009, wires=[9,10])
    qml.CRX(x010, wires=[10,11])
    qml.CRX(x011, wires=[11,12])
    qml.CRX(x012, wires=[12,13])
    qml.CRX(x013, wires=[13,14])
    qml.CRX(x014, wires=[14,15])
    qml.CRX(x015, wires=[15,0])

    for rx in range(16):
      qml.RX(inputs[rx], wires=rx)
    qml.Rot(*w100, wires=0)
    qml.Rot(*w101, wires=1)
    qml.Rot(*w102, wires=2)
    qml.Rot(*w103, wires=3)
    qml.Rot(*w104, wires=4)
    qml.Rot(*w105, wires=5)
    qml.Rot(*w106, wires=6)
    qml.Rot(*w107, wires=7)
    qml.Rot(*w108, wires=8)
    qml.Rot(*w109, wires=9)
    qml.Rot(*w110, wires=10)
    qml.Rot(*w111, wires=11)
    qml.Rot(*w112, wires=12)
    qml.Rot(*w113, wires=13)
    qml.Rot(*w114, wires=14)
    qml.Rot(*w115, wires=15)
    
    qml.CRX(x100, wires=[0,1])
    qml.CRX(x101, wires=[1,2])
    qml.CRX(x102, wires=[2,3])
    qml.CRX(x103, wires=[3,4])
    qml.CRX(x104, wires=[4,5])
    qml.CRX(x105, wires=[5,6])
    qml.CRX(x106, wires=[6,7])
    qml.CRX(x107, wires=[7,8])
    qml.CRX(x108, wires=[8,9])
    qml.CRX(x109, wires=[9,10])
    qml.CRX(x110, wires=[10,11])
    qml.CRX(x111, wires=[11,12])
    qml.CRX(x112, wires=[12,13])
    qml.CRX(x113, wires=[13,14])
    qml.CRX(x114, wires=[14,15])
    qml.CRX(x115, wires=[15,0])


    return (qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3)), qml.expval(qml.PauliZ(4)),
    qml.expval(qml.PauliZ(5)), qml.expval(qml.PauliZ(6)), qml.expval(qml.PauliZ(7)), qml.expval(qml.PauliZ(8)), qml.expval(qml.PauliZ(9)), 
    qml.expval(qml.PauliZ(10)), qml.expval(qml.PauliZ(11)), qml.expval(qml.PauliZ(12)), qml.expval(qml.PauliZ(13)), qml.expval(qml.PauliZ(14)), 
    qml.expval(qml.PauliZ(15)))

weight_shapes = {"w000": 3, "w001": 3, "w002": 3, "w003": 3, "w004": 3, "w005": 3, "w006": 3, "w007": 3, 
          "w008": 3, "w009": 3, "w010": 3, "w011": 3, "w012": 3, "w013": 3, "w014": 3, "w015": 3, 
          "x000": 1, "x001": 1, "x002": 1, "x003": 1, "x004": 1, "x005": 1, "x006": 1, "x007": 1, 
          "x008": 1, "x009": 1, "x010": 1, "x011": 1, "x012": 1, "x013": 1, "x014": 1, "x015": 1, 
          "w100": 3, "w101": 3, "w102": 3, "w103": 3, "w104": 3, "w105": 3, "w106": 3, "w107": 3, 
          "w108": 3, "w109": 3, "w110": 3, "w111": 3, "w112": 3, "w113": 3, "w114": 3, "w115": 3,
          "x100": 1, "x101": 1, "x102": 1, "x103": 1, "x104": 1, "x105": 1, "x106": 1, "x107": 1, 
          "x108": 1, "x109": 1, "x110": 1, "x111": 1, "x112": 1, "x113": 1, "x114": 1, "x115": 1}


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
plt.show()
