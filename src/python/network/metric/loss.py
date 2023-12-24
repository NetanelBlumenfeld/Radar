from enum import Enum

import torch
import torch.nn as nn


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
        compute loss for tiny radar classifier

        Args:
            outputs (torch.Tensor): the outputs from TinyRadarNN model
            labels (torch.Tensor): labels for the data

        Returns:
            loss (float): the loss
        """
        return self.loss_function(outputs, labels)


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
            loss += self.loss_function(outputs[i], labels[i]) * 100
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
