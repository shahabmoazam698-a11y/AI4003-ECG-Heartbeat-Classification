
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import ECGNet

# Load trained model
project_dir = Path(__file__).resolve().parent.parent
model_path = project_dir / "results" / "best_model.pth"

model = ECGNet()
state_dict = torch.load(
    model_path,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)

# Display model weights as heatmaps
weights = [
    ("W1: Input to Hidden Layer 1", model.W1.detach().numpy()),
    ("W2: Hidden Layer 1 to Hidden Layer 2", model.W2.detach().numpy()),
    ("W3: Hidden Layer 2 to Output", model.W3.detach().numpy())
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (title, weight) in zip(axes, weights):
    image = ax.imshow(weight, cmap="coolwarm", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Output neurons")
    ax.set_ylabel("Input neurons")
    plt.colorbar(image, ax=ax)

plt.tight_layout()

output_path = project_dir / "results" / "figures" / "model_weights.png"
plt.savefig(output_path, dpi=300)
plt.show()

print("Weight visualization saved at:", output_path)
