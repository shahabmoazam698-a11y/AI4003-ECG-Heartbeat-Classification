
import math
import torch
import torch.nn as nn


class ECGNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Input layer: 187 -> 128
        self.W1 = nn.Parameter(
            torch.randn(187, 128) * math.sqrt(2.0 / 187)
        )
        self.b1 = nn.Parameter(torch.zeros(128))

        # Hidden layer: 128 -> 64
        self.W2 = nn.Parameter(
            torch.randn(128, 64) * math.sqrt(2.0 / 128)
        )
        self.b2 = nn.Parameter(torch.zeros(64))

        # Output layer: 64 -> 5
        self.W3 = nn.Parameter(
            torch.randn(64, 5) * math.sqrt(2.0 / 64)
        )
        self.b3 = nn.Parameter(torch.zeros(5))

    def forward(self, x):

        z1 = x @ self.W1 + self.b1
        a1 = torch.clamp(z1, min=0)

        z2 = a1 @ self.W2 + self.b2
        a2 = torch.clamp(z2, min=0)

        logits = a2 @ self.W3 + self.b3

        return logits


if __name__ == "__main__":

    torch.manual_seed(0)

    model = ECGNet()

    sample_input = torch.randn(10, 187)

    output = model(sample_input)

    total_parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("Neural network created successfully!")
    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)
    print("Trainable parameters:", total_parameters)
