
from pathlib import Path
import matplotlib.pyplot as plt

from prepare import prepare_data


def plot_examples():

    # Load the prepared dataset
    data = prepare_data()

    X_train = data["X_train"]
    y_train = data["y_train"]

    # Class names
    class_names = {
        0: "Normal (N)",
        1: "Supraventricular ectopic (S)",
        2: "Ventricular ectopic (V)",
        3: "Fusion (F)",
        4: "Unknown / paced (Q)"
    }

    # Create a figure containing five plots
    fig, axes = plt.subplots(5, 1, figsize=(12, 14))

    for class_id in range(5):

        # Find the first training example belonging to this class
        indices = (y_train == class_id).nonzero(as_tuple=True)[0]

        if len(indices) == 0:
            print(f"No example found for class {class_id}")
            continue

        # Select one ECG heartbeat
        signal = X_train[indices[0]].numpy()

        # Plot the 187 signal measurements
        axes[class_id].plot(range(187), signal)

        axes[class_id].set_title(
            f"Class {class_id}: {class_names[class_id]}"
        )

        axes[class_id].set_xlabel("Sample number")
        axes[class_id].set_ylabel("Amplitude")
        axes[class_id].grid(True)

    plt.tight_layout()

    # Save the figure
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "ecg_class_examples.png"

    plt.savefig(output_path, dpi=300)
    print(f"\nFigure saved to: {output_path}")

    # Display the figure
    plt.show()


if __name__ == "__main__":
    plot_examples()
