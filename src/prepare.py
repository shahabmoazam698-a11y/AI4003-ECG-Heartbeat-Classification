
import numpy as np
import torch

from load_data import load_raw


def prepare_data(data_dir="data"):
    """
    Separate ECG signals from labels,
    split training data into training and validation sets,
    and convert everything into PyTorch tensors.
    """

    # Step 1: Load the original CSV files
    raw = load_raw(data_dir)

    train_df = raw["train"]
    test_df = raw["test"]

    # Step 2: Separate input signals (X) and labels (y)
    X_full = train_df.iloc[:, :187].to_numpy(dtype=np.float32)
    y_full = train_df.iloc[:, 187].to_numpy(dtype=np.int64)

    X_test = test_df.iloc[:, :187].to_numpy(dtype=np.float32)
    y_test = test_df.iloc[:, 187].to_numpy(dtype=np.int64)

    # Step 3: Create the required training/validation split
    rng = np.random.default_rng(42)

    idx = rng.permutation(len(train_df))

    n_val = int(0.1 * len(train_df))

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    # Step 4: Select training and validation records
    X_train = X_full[train_idx]
    y_train = y_full[train_idx]

    X_val = X_full[val_idx]
    y_val = y_full[val_idx]

    # Step 5: Convert NumPy arrays into PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }


if __name__ == "__main__":

    data = prepare_data()

    print("\nDataset preparation completed!\n")

    for name, tensor in data.items():
        print(
            f"{name}: shape = {tuple(tensor.shape)}, "
            f"dtype = {tensor.dtype}"
        )

    print("\nTraining class distribution:")

    classes, counts = torch.unique(
        data["y_train"],
        return_counts=True
    )

    for class_id, count in zip(classes, counts):
        print(f"Class {class_id.item()}: {count.item()} records")
