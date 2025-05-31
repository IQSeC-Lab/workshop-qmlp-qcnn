from collections import Counter
import numpy as np
import os




def trim_to_top_classes_and_save(name, X_train, y_train, X_test, y_test, top_n, output_dir='.'):
  
    # Count class frequencies
    class_counts = Counter(y_train)
    top_classes = [cls for cls, _ in class_counts.most_common(top_n)]

    # Create masks
    train_mask = np.isin(y_train, top_classes)
    test_mask = np.isin(y_test, top_classes)

    # Filter data
    X_train_filtered = X_train[train_mask]
    y_train_filtered = y_train[train_mask]
    X_test_filtered = X_test[test_mask]
    y_test_filtered = y_test[test_mask]

    # Remap class labels to 0...(top_n-1)
    unique_classes = np.unique(y_train_filtered)
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_train_mapped = np.array([class_to_idx[y] for y in y_train_filtered])
    y_test_mapped = np.array([class_to_idx[y] for y in y_test_filtered])

    # Save to file

    # Train
    output_path = os.path.join(output_dir, f"{name}_{top_n}_families_train.npz")
    np.savez_compressed(output_path, X_train=X_train_filtered, Y_train=y_train_mapped)
    
    # Test
    output_path1 = os.path.join(output_dir, f"{name}_{top_n}_families_test.npz")
    np.savez_compressed(output_path1,X_test=X_test_filtered, Y_test=y_test_mapped)
    
    print(f"Saved trimmed dataset to {output_path}")
    print(f"Saved trimmed dataset to {output_path1}")

data_dir = "../dataset/Ember-Class-100class"
name = "Ember-Class-100class"
train_file = os.path.join(data_dir, 'XY_train.npz')
test_file = os.path.join(data_dir, 'XY_test.npz')
 
# 1. Load train and test splits
ember_train = np.load(train_file)
ember_test = np.load(test_file)

X_train = ember_train["X_train"]
y_train = ember_train["Y_train"]
X_test = ember_test["X_test"]
y_test = ember_test["Y_test"]
# Example usage
trim_to_top_classes_and_save(name, X_train, y_train, X_test, y_test, top_n=14)

