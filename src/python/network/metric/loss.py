import torch


class LossFunctionTinyRadarNN:
    def __init__(
        self,
        numberOfTimeSteps: int,
        loss_function: callable = torch.nn.CrossEntropyLoss(),
    ):
        self.loss_function = loss_function
        self.numberOfTimeSteps = numberOfTimeSteps

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


class LossFunctionSRCnnTinyRadarNN:
    def __init__(
        self,
        loss_func_srcnn: callable = torch.nn.MSELoss(),
        loss_func_classifier: callable = torch.nn.CrossEntropyLoss(),
        wight_srcnn: float = 0.5,
        wight_classifier: float = 0.5,
    ):
        self.loss_func_srcnn = loss_func_srcnn
        self.loss_func_classifier = loss_func_classifier
        self.wight_srcnn = wight_srcnn
        self.wight_classifier = wight_classifier

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
