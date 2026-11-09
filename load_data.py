
from pathlib import Path
import pandas as pd


# Names of the two dataset files
FILES = {
    "train": "mitbih_train.csv",
    "test": "mitbih_test.csv"
}


def load_raw(data_dir="data"):
    """
    Load the original training and testing CSV files.
    Each row contains 187 ECG signal values and 1 class label.
    """

    data_dir = Path(data_dir)
    frames = {}

    for split, filename in FILES.items():

        file_path = data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}\n"
                "Check that the CSV files are inside the data folder."
            )

        frames[split] = pd.read_csv(file_path, header=None)

    return frames


if __name__ == "__main__":

    data = load_raw()

    for split, df in data.items():
        print(f"{split}: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nFirst three training rows:")
    print(data["train"].head(3))

    print("\nClass distribution in training data:")
    print(data["train"].iloc[:, 187].value_counts().sort_index())
