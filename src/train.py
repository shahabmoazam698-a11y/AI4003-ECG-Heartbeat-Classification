
import sys
import math
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

sys.path.insert(0, str(SRC_DIR))

from prepare import prepare_data
from model import ECGNet
from loss import manual_cross_entropy


# Training settings
BATCH_SIZE = 128
EPOCHS = 30
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

LEARNING_RATES = [0.01, 0.1, 0.5]


def calculate_metrics(model, X, y):
    """Calculate loss and accuracy without updating parameters."""

    model.eval()

    with torch.no_grad():
        logits = model(X)
        loss = manual_cross_entropy(logits, y).item()
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean().item()

    return loss, accuracy


def save_checkpoint(path, model, velocities, history,
                    best_state, best_loss, best_epoch, epoch):

    checkpoint = {
        "model_state": model.state_dict(),
        "velocities": velocities,
        "history": history,
        "best_state": best_state,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "epoch": epoch
    }

    torch.save(checkpoint, path)


def train_one_model(X_train, y_train, X_val, y_val, learning_rate):

    checkpoint_dir = PROJECT_DIR / "results" / "checkpoints"
    tables_dir = PROJECT_DIR / "results" / "tables"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        checkpoint_dir / f"checkpoint_lr_{learning_rate}.pth"
    )

    csv_path = tables_dir / f"history_lr_{learning_rate}.csv"

    # Start a fresh model with the required seed
    torch.manual_seed(0)
    model = ECGNet()

    velocities = {
        name: torch.zeros_like(parameter)
        for name, parameter in model.named_parameters()
    }

    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    start_epoch = 0

    # Resume if a checkpoint already exists
    if checkpoint_path.exists():

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(checkpoint["model_state"])
        velocities = checkpoint["velocities"]
        history = checkpoint["history"]
        best_state = checkpoint["best_state"]
        best_loss = checkpoint["best_loss"]
        best_epoch = checkpoint["best_epoch"]
        start_epoch = checkpoint["epoch"]

        print(f"\nResuming LR={learning_rate} from epoch "
              f"{start_epoch + 1}")

    else:
        print(f"\nStarting LR={learning_rate} from epoch 1")

    # Skip an experiment that already completed
    if start_epoch >= EPOCHS:
        print(f"LR={learning_rate} already completed.")
        return model, history, best_loss, best_epoch

    for epoch in range(start_epoch, EPOCHS):

        model.train()

        # Shuffle training samples
        indices = torch.randperm(len(X_train))

        for start in range(0, len(X_train), BATCH_SIZE):

            batch_indices = indices[start:start + BATCH_SIZE]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Forward pass
            logits = model(X_batch)

            # Manual cross-entropy loss
            loss = manual_cross_entropy(logits, y_batch)

            # Calculate gradients
            model.zero_grad()
            loss.backward()

            # Manual momentum and weight decay update
            with torch.no_grad():

                for name, parameter in model.named_parameters():

                    gradient = parameter.grad

                    if name in ["W1", "W2", "W3"]:
                        gradient = gradient + WEIGHT_DECAY * parameter

                    velocities[name] = (
                        MOMENTUM * velocities[name]
                        + (1 - MOMENTUM) * gradient
                    )

                    parameter -= learning_rate * velocities[name]

        # Evaluate after completing the epoch
        train_loss, train_acc = calculate_metrics(
            model, X_train, y_train
        )

        val_loss, val_acc = calculate_metrics(
            model, X_val, y_val
        )

        # Stop safely if this learning rate diverges
        if not all(map(math.isfinite, [
            train_loss, train_acc, val_loss, val_acc
        ])):
            print(f"LR={learning_rate} diverged at epoch {epoch + 1}.")
            break

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        }

        history.append(record)

        print(
            f"LR={learning_rate} | "
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Keep the best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }

        # Save history after every epoch
        pd.DataFrame(history).to_csv(csv_path, index=False)

        # Save checkpoint after every completed epoch
        save_checkpoint(
            checkpoint_path,
            model,
            velocities,
            history,
            best_state,
            best_loss,
            best_epoch,
            epoch + 1
        )

    return model, history, best_loss, best_epoch


def main():

    torch.set_num_threads(2)

    print("Loading ECG dataset...")

    data = prepare_data()

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    results_dir = PROJECT_DIR / "results"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    comparison = []

    global_best_loss = float("inf")
    global_best_state = None
    global_best_lr = None
    global_best_epoch = None

    for learning_rate in LEARNING_RATES:

        print("\n" + "=" * 60)
        print(f"Training with learning rate: {learning_rate}")
        print("=" * 60)

        model, history, best_loss, best_epoch = train_one_model(
            X_train,
            y_train,
            X_val,
            y_val,
            learning_rate
        )

        checkpoint_path = (
            results_dir / "checkpoints"
            / f"checkpoint_lr_{learning_rate}.pth"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        best_state = checkpoint["best_state"]

        comparison.append({
            "learning_rate": learning_rate,
            "best_validation_loss": best_loss,
            "best_epoch": best_epoch,
            "epochs_completed": len(history)
        })

        # Select the best learning rate
        if best_state is not None and best_loss < global_best_loss:

            global_best_loss = best_loss
            global_best_state = best_state
            global_best_lr = learning_rate
            global_best_epoch = best_epoch

            torch.save(
                global_best_state,
                results_dir / "best_model.pth"
            )

            print("New best model saved!")

    # Save learning-rate comparison
    comparison_df = pd.DataFrame(comparison)

    comparison_df.to_csv(
        tables_dir / "learning_rate_comparison.csv",
        index=False
    )

    if global_best_state is None:
        print("\nNo valid model was produced. Check training results.")
        return

    # Save final model
    torch.save(
        global_best_state,
        results_dir / "best_model.pth"
    )

    # Save training summary
    with open(results_dir / "training_summary.txt", "w") as file:
        file.write("ECG TRAINING SUMMARY\n\n")
        file.write(f"Best learning rate: {global_best_lr}\n")
        file.write(f"Best epoch: {global_best_epoch}\n")
        file.write(f"Best validation loss: {global_best_loss:.6f}\n")

    # Plot validation loss comparison
    plt.figure(figsize=(8, 5))

    for learning_rate in LEARNING_RATES:
        csv_path = tables_dir / f"history_lr_{learning_rate}.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                plt.plot(
                    df["epoch"],
                    df["val_loss"],
                    label=f"LR={learning_rate}"
                )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss for Different Learning Rates")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        figures_dir / "learning_rate_comparison.png",
        dpi=300
    )
    plt.close()

    # Plot best model training and validation curves
    best_history_path = (
        tables_dir / f"history_lr_{global_best_lr}.csv"
    )

    best_history = pd.read_csv(best_history_path)

    plt.figure(figsize=(8, 5))
    plt.plot(best_history["epoch"], best_history["train_loss"],
             label="Training Loss")
    plt.plot(best_history["epoch"], best_history["val_loss"],
             label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Best Model: Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "best_model_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(best_history["epoch"], best_history["train_accuracy"],
             label="Training Accuracy")
    plt.plot(best_history["epoch"], best_history["val_accuracy"],
             label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Best Model: Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_dir / "best_model_accuracy.png", dpi=300)
    plt.close()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best learning rate: {global_best_lr}")
    print(f"Best epoch: {global_best_epoch}")
    print(f"Best validation loss: {global_best_loss:.6f}")
    print("Best model saved to results/best_model.pth")


if __name__ == "__main__":
    main()
