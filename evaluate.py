
import sys
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from prepare import prepare_data
from model import ECGNet


def main():
    # Load prepared training, validation and test data
    data = prepare_data()

    X_test = data["X_test"]
    y_test = data["y_test"]

    # Load the best trained model
    model = ECGNet()

    model_path = PROJECT_DIR / "results" / "best_model.pth"
    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    # Make predictions
    with torch.no_grad():
        logits = model(X_test)
        predictions = torch.argmax(logits, dim=1)

    # Overall test accuracy
    correct = (predictions == y_test).sum().item()
    total = len(y_test)
    test_accuracy = correct / total

    # Create confusion matrix
    confusion = torch.zeros((5, 5), dtype=torch.int64)

    for actual, predicted in zip(y_test, predictions):
        confusion[actual.item(), predicted.item()] += 1

    # Calculate accuracy for each class
    class_names = ["N", "S", "V", "F", "Q"]
    class_accuracies = []

    for i in range(5):
        class_total = confusion[i].sum().item()
        class_correct = confusion[i, i].item()

        accuracy = class_correct / class_total
        class_accuracies.append(accuracy)

    # Balanced accuracy
    balanced_accuracy = sum(class_accuracies) / 5

    # Majority-class baseline: always predict N (class 0)
    majority_correct = (y_test == 0).sum().item()
    majority_accuracy = majority_correct / total

    # Print results
    print("\n========== TEST RESULTS ==========")
    print(f"Test samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Balanced accuracy: {balanced_accuracy * 100:.2f}%")
    print(f"Majority baseline accuracy: {majority_accuracy * 100:.2f}%")

    print("\n========== CONFUSION MATRIX ==========")
    print("Rows = Actual classes")
    print("Columns = Predicted classes")
    print(confusion.numpy())

    print("\n========== PER-CLASS ACCURACY ==========")
    for name, accuracy in zip(class_names, class_accuracies):
        print(f"{name}: {accuracy * 100:.2f}%")

    # Save results folder
    results_dir = PROJECT_DIR / "results"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix as CSV
    confusion_df = pd.DataFrame(
        confusion.numpy(),
        index=class_names,
        columns=class_names
    )

    confusion_df.to_csv(tables_dir / "confusion_matrix.csv")

    # Save per-class accuracy as CSV
    accuracy_df = pd.DataFrame({
        "Class": class_names,
        "Accuracy": class_accuracies
    })

    accuracy_df.to_csv(
        tables_dir / "per_class_accuracy.csv",
        index=False
    )

    # Save text report
    report_path = results_dir / "test_results.txt"

    with open(report_path, "w") as file:
        file.write("ECG HEARTBEAT CLASSIFICATION - TEST RESULTS\n\n")
        file.write(f"Test samples: {total}\n")
        file.write(f"Correct predictions: {correct}\n")
        file.write(f"Test accuracy: {test_accuracy * 100:.2f}%\n")
        file.write(
            f"Balanced accuracy: {balanced_accuracy * 100:.2f}%\n"
        )
        file.write(
            f"Majority baseline accuracy: {majority_accuracy * 100:.2f}%\n\n"
        )

        file.write("Confusion Matrix:\n")
        file.write(str(confusion.numpy()))
        file.write("\n\nPer-class accuracy:\n")

        for name, accuracy in zip(class_names, class_accuracies):
            file.write(f"{name}: {accuracy * 100:.2f}%\n")

    # Plot confusion matrix
    plt.figure(figsize=(7, 6))
    plt.imshow(confusion.numpy(), cmap="Blues")
    plt.colorbar()

    plt.xticks(range(5), class_names)
    plt.yticks(range(5), class_names)

    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("ECG Test Confusion Matrix")

    for i in range(5):
        for j in range(5):
            plt.text(
                j, i,
                str(confusion[i, j].item()),
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.savefig(
        figures_dir / "confusion_matrix.png",
        dpi=300
    )
    plt.close()

    print("\nResults saved successfully!")


if __name__ == "__main__":
    main()
