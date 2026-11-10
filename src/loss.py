
import sys
from pathlib import Path
import torch

# Add the src folder to Python's import path
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from prepare import prepare_data
from model import ECGNet


def manual_cross_entropy(logits, targets):
    """
    Manually calculate softmax cross-entropy loss.

    logits: model outputs, shape (batch_size, 5)
    targets: correct class labels, shape (batch_size,)
    """

    # Select the logit corresponding to the correct class
    correct_logits = logits.gather(
        dim=1,
        index=targets.reshape(-1, 1)
    ).squeeze(1)

    # Calculate numerically stable log-sum-exp
    log_sum_exp = torch.logsumexp(logits, dim=1)

    # Calculate individual losses
    losses = -correct_logits + log_sum_exp

    # Return mean loss
    return losses.mean()


def main():

    # Load prepared dataset
    data = prepare_data()

    X_train = data["X_train"]
    y_train = data["y_train"]

    # Set random seed before creating model
    torch.manual_seed(0)

    # Create neural network
    model = ECGNet()

    # Select first 64 training examples
    X_batch = X_train[:64]
    y_batch = y_train[:64]

    # Forward pass
    logits = model(X_batch)

    # Calculate manual cross-entropy loss
    loss = manual_cross_entropy(logits, y_batch)

    # Display results
    print("Manual cross-entropy loss implemented successfully!")
    print("-----------------------------------------------")
    print("Input shape:", X_batch.shape)
    print("Logits shape:", logits.shape)
    print("Targets shape:", y_batch.shape)
    print("Loss:", loss.item())

    # Save results
    output_dir = SRC_DIR.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "loss_info.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("Manual Softmax Cross-Entropy Loss\n")
        file.write("=================================\n")
        file.write(f"Input shape: {tuple(X_batch.shape)}\n")
        file.write(f"Logits shape: {tuple(logits.shape)}\n")
        file.write(f"Targets shape: {tuple(y_batch.shape)}\n")
        file.write(f"Loss: {loss.item():.8f}\n")

    print("\nLoss information saved successfully!")
    print("Saved at:", output_file)


if __name__ == "__main__":
    main()
