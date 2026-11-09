
import sys
from pathlib import Path
import torch

# Add src folder to Python import path
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from prepare import prepare_data
from model import ECGNet
from loss import manual_cross_entropy


def main():

    # Step 1: Set random seed
    torch.manual_seed(0)

    # Step 2: Load dataset
    data = prepare_data()

    # Use the first 64 training examples
    X_train = data["X_train"][:64]
    y_train = data["y_train"][:64]

    batch_size = X_train.shape[0]

    # Step 3: Create model
    model = ECGNet()

    # Clear gradients
    model.zero_grad()

    # Step 4: Forward pass
    z1 = X_train @ model.W1 + model.b1
    a1 = torch.clamp(z1, min=0)

    z2 = a1 @ model.W2 + model.b2
    a2 = torch.clamp(z2, min=0)

    logits = a2 @ model.W3 + model.b3

    # Step 5: Calculate loss
    loss = manual_cross_entropy(logits, y_train)

    # Step 6: Calculate automatic gradients
    loss.backward()

    # Step 7: Calculate softmax probabilities
    shifted_logits = logits.detach() - logits.detach().max(
        dim=1, keepdim=True
    ).values

    exp_logits = torch.exp(shifted_logits)

    probabilities = exp_logits / exp_logits.sum(
        dim=1, keepdim=True
    )

    # Step 8: Create one-hot encoded labels
    Y = torch.zeros_like(probabilities)

    Y.scatter_(
        dim=1,
        index=y_train.reshape(-1, 1),
        value=1.0
    )

    # Step 9: Calculate output error
    error = probabilities - Y

    # Step 10: Calculate manual W3 gradient
    # Correct shape: (64, 5)
    manual_dW3 = (a2.detach().T @ error) / batch_size

    # Step 11: Calculate manual b3 gradient
    # Correct shape: (5,)
    manual_db3 = error.sum(dim=0) / batch_size

    # Step 12: Get automatic gradients
    automatic_dW3 = model.W3.grad
    automatic_db3 = model.b3.grad

    # Step 13: Verify gradient dimensions
    print("\nGRADIENT DIMENSIONS")
    print("===================")

    print("a2 shape:", a2.shape)
    print("Error shape:", error.shape)
    print("Manual W3 shape:", manual_dW3.shape)
    print("Automatic W3 shape:", automatic_dW3.shape)
    print("Manual b3 shape:", manual_db3.shape)
    print("Automatic b3 shape:", automatic_db3.shape)

    assert manual_dW3.shape == automatic_dW3.shape
    assert manual_db3.shape == automatic_db3.shape

    # Step 14: Calculate maximum absolute differences
    difference_W3 = torch.max(
        torch.abs(manual_dW3 - automatic_dW3)
    ).item()

    difference_b3 = torch.max(
        torch.abs(manual_db3 - automatic_db3)
    ).item()

    # Step 15: Check tolerance
    tolerance = 1e-5

    passed = (
        difference_W3 < tolerance
        and difference_b3 < tolerance
    )

    # Step 16: Display results
    print("\nGRADIENT CHECK RESULTS")
    print("======================")

    print("Batch size:", batch_size)
    print("Loss:", loss.item())

    print(f"\nW3 maximum difference: {difference_W3:.12f}")
    print(f"b3 maximum difference: {difference_b3:.12f}")

    print("\nTolerance:", tolerance)

    if passed:
        print("\nRESULT: GRADIENT CHECK PASSED")
    else:
        print("\nRESULT: GRADIENT CHECK FAILED")

    # Step 17: Save results
    output_dir = SRC_DIR.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "gradient_check.txt"

    with open(output_file, "w", encoding="utf-8") as file:

        file.write("GRADIENT CHECK RESULTS\n")
        file.write("======================\n")

        file.write(f"Batch size: {batch_size}\n")
        file.write(f"Loss: {loss.item():.12f}\n")

        file.write(f"W3 gradient difference: {difference_W3:.12f}\n")
        file.write(f"b3 gradient difference: {difference_b3:.12f}\n")

        file.write(f"Tolerance: {tolerance}\n")

        if passed:
            file.write("RESULT: GRADIENT CHECK PASSED\n")
        else:
            file.write("RESULT: GRADIENT CHECK FAILED\n")

    print("\nResults saved successfully!")
    print("File:", output_file)


if __name__ == "__main__":
    main()
