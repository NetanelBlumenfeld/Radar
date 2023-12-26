from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossType(Enum):
    L1 = "L1"
    MSE = "MSE"
    CrossEntropy = "CrossEntropy"
    Huber = "Huber"


class LossFactory:
    loss_functions = {
        "l1": nn.L1Loss(),
        "mse": nn.MSELoss(),
        "huber": nn.HuberLoss(),
        "crossentropy": nn.CrossEntropyLoss(),
    }

    @staticmethod
    def get_loss_function(name):
        if name not in LossFactory.loss_functions:
            raise ValueError(f"Loss function '{name}' not recognized")
        return LossFactory.loss_functions[name]


class SimpleLoss:
    def __init__(self, loss_function: LossType):
        self.loss_function = LossFactory.get_loss_function(loss_function.name.lower())
        self.name = loss_function.name

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        compute loss

        Args:
            outputs (torch.Tensor): the outputs from TinyRadarNN model
            labels (torch.Tensor): labels for the data

        Returns:
            loss (float): the loss
        """
        return self.loss_function(outputs, labels) * 15


class LossFunctionTinyRadarNN:
    def __init__(
        self,
        loss_function: LossType,
        numberOfTimeSteps: int = 5,
    ):
        self.numberOfTimeSteps = numberOfTimeSteps
        self.loss_function = LossFactory.get_loss_function(loss_function.name.lower())
        self.name = loss_function.name

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        compute loss for tiny radar classifier

        Args:
            outputs (torch.Tensor): the outputs from TinyRadarNN model
            labels (torch.Tensor): labels for the data

        Returns:
            loss (float): the loss
        """
        loss = 0
        for i in range(self.numberOfTimeSteps):
            loss += self.loss_function(outputs[i], labels[i])
        return loss / self.numberOfTimeSteps


class LossFunctionSRTinyRadarNN:
    def __init__(
        self,
        loss_type_srcnn: LossType,
        loss_type_classifier: LossType,
        wight_srcnn: float = 0.5,
        wight_classifier: float = 0.5,
    ):
        self.loss_func_srcnn = LossFunctionTinyRadarNN(loss_type_srcnn)
        self.loss_func_classifier = LossFunctionTinyRadarNN(loss_type_classifier)
        self.wight_srcnn = wight_srcnn
        self.wight_classifier = wight_classifier
        self.name = f"sr_{loss_type_srcnn.name}_classifier_{loss_type_classifier.name}"

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        compute loss for tiny radar classifier

        Args:
            outputs (torch.Tensor): the outputs from TinyRadarNN model
            labels (torch.Tensor): labels for the data

        Returns:
            loss (float): the loss
        """
        loss_srcnn = self.loss_func_srcnn(outputs[0], labels[0])
        loss_classifier = self.loss_func_classifier(outputs[1], labels[1])
        loss = self.wight_srcnn * loss_srcnn + self.wight_classifier * loss_classifier
        return loss, loss_classifier, loss_srcnn


class SSIMLoss(nn.Module):
    def __init__(self, C1=0.01**2, C2=0.03**2, sigma=5.0):
        super(SSIMLoss, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.sigma = sigma

    def gaussian_filter(self, channel, kernel_size):
        kernel = torch.range(-(kernel_size // 2), kernel_size // 2, dtype=torch.float32)
        kernel = torch.exp(-0.5 * kernel**2 / self.sigma**2)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(channel, 1, 1, 1)
        return kernel

    def forward(self, input1, input2):
        # Check dimensions
        if input1.size() != input2.size():
            raise ValueError("Inputs must have the same dimension.")

        # Gaussian filter
        channel = input1.size(1)
        kernel_size = int(2 * round(3 * self.sigma) + 1)
        if kernel_size % 2 == 0:
            raise ValueError("Odd kernel size preferred")
        gaussian_kernel = self.gaussian_filter(channel, kernel_size)

        # Ensure the input tensor is on the same device as the kernel
        gaussian_kernel = gaussian_kernel.to(input1.device)

        # Convolution
        padding = kernel_size // 2
        mux = F.conv2d(input1, gaussian_kernel, padding=padding, groups=channel)
        muy = F.conv2d(input2, gaussian_kernel, padding=padding, groups=channel)

        sigmax2 = (
            F.conv2d(input1 * input1, gaussian_kernel, padding=padding, groups=channel)
            - mux**2
        )
        sigmay2 = (
            F.conv2d(input2 * input2, gaussian_kernel, padding=padding, groups=channel)
            - muy**2
        )
        sigmaxy = (
            F.conv2d(input1 * input2, gaussian_kernel, padding=padding, groups=channel)
            - mux * muy
        )

        l = (2 * mux * muy + self.C1) / (mux**2 + muy**2 + self.C1)
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        ssim = l * cs
        return 1 - ssim.mean()
