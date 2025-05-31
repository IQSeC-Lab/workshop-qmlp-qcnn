import numpy as np

# Load the dataset (update paths if needed)
data_train = np.load("../dataset/AZ-Class-Task/AZ-Class-Task_23_families_train.npz")
data_test = np.load("../dataset/AZ-Class-Task/AZ-Class-Task_23_families_test.npz")

# Extract
X_train = data_train["X_train"]
y_train = data_train["Y_train"]
X_test = data_test["X_test"]
y_test = data_test["Y_test"]

# Print basic info
print("TRAIN SET")
print("X_train shape:", X_train.shape)
print("Y_train shape:", y_train.shape)
print("Y_train head:")
print(y_train[:5])

print("\nTEST SET")
print("X_test shape:", X_test.shape)
print("Y_test shape:", y_test.shape)
print("Y_test head:")
print(y_test[:5])

# Optional: Check if labels are one-hot
def check_one_hot(y):
    if y.ndim == 2 and ((y == 0) | (y == 1)).all() and (y.sum(axis=1) == 1).all():
        return True
    return False

print("\nLabel format:")
print("Y_train is one-hot encoded:", check_one_hot(y_train))
print("Y_test is one-hot encoded:", check_one_hot(y_test))
