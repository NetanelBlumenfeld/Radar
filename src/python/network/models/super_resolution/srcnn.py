import torch
import torch.nn as nn
from network.models.basic_model import BasicModel


class SRCnn(BasicModel):
    def __init__(
        self,
        num_channels=1,
        num_features_1=64,
        num_features_2=64,
        kernel_size=(3, 3),
        activation="relu",
        base_name: str = "SRCnn",
        only_wights: bool = False,
    ):
        model_name = (
            base_name
            + "_features_1_"
            + str(num_features_1)
            + "_features_2_"
            + str(num_features_2)
            + "_k_size_"
            + str(kernel_size[0])
            + "_activation_"
            + str(activation)
        )
        super(SRCnn, self).__init__(model_name, only_wights)
        # Define the activation function using a dictionary
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "gelu": nn.GELU(),
        }

        self.activation = activations.get(activation)
        if self.activation is None:
            raise ValueError(f"Invalid activation function: {activation}")

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

    @staticmethod
    def reshape_to_model_output(low_res, high_res, device):
        batch_size, time_steps, channels = (
            low_res.shape[0],
            low_res.shape[1],
            low_res.shape[2],
        )
        X = low_res.reshape(
            batch_size * time_steps * channels, 1, low_res.shape[3], low_res.shape[4]
        )
        y = high_res.reshape(
            batch_size * time_steps * channels, 1, high_res.shape[3], high_res.shape[4]
        )
        return X.to(device), y.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.activation is None:
            raise ValueError("Activation function is not properly initialized.")
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x)) + identity
        return x
