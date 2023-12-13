import torch
import torch.nn as nn


class SRCnn(nn.Module):
    def __init__(self, num_channels=2, activation="relu", k_size=3):
        super(SRCnn, self).__init__()

        # Define the activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        # Add support for other activation functions if needed

        # Define the layers
        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=k_size, padding=k_size // 2
        )
        self.conv2 = nn.Conv2d(64, 32, kernel_size=k_size, padding=k_size // 2)
        self.conv3 = nn.Conv2d(
            32, num_channels, kernel_size=k_size, padding=k_size // 2
        )

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        return x
