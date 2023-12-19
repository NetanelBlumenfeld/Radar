import torch.nn as nn


class SRCnn(nn.Module):
    def __init__(
        self,
        num_channels=1,
        num_features_1=64,
        num_features_2=64,
        kernel_size=(3, 3),
        activation="relu",
    ):
        super(SRCnn, self).__init__()

        # Define the activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("Invalid activation function")

        # Define the layers with hyperparameters
        self.conv1 = nn.Conv2d(
            num_channels,
            num_features_1,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        )
        self.conv2 = nn.Conv2d(
            num_features_1,
            num_features_2,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        )
        self.conv3 = nn.Conv2d(
            num_features_1,
            num_channels,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        )
        self.bn1 = nn.BatchNorm2d(num_features_1)
        self.bn2 = nn.BatchNorm2d(num_features_2)

    def forward(self, x):
        identity = x
        x = self.activation(self.bn1(self.conv1(x)))
        # x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.conv3(x)) + identity
        return x
