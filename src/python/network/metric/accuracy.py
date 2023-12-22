import torch


def acc_tiny_radar(
    outputs: torch.Tensor, labels: torch.Tensor, numberOfGestures: int = 12
):
    """
    compute accuracy for tiny radar classifier

    Args:
        outputs (torch.Tensor): the outputs from TinyRadarNN model
        labels (torch.Tensor): labels for the data
        numberOfGestures (int): number of classes

    Returns:
        correct (float): the number of correct predictions
        total (float): the total number of predictions
    """
    pred = outputs.reshape(-1, numberOfGestures).max(1)
    squashed_labels = labels.reshape(-1)
    total = squashed_labels.shape[0]
    correct = pred[1].eq(squashed_labels).sum().item()
    return total, correct


def acc_srcnn_tiny_radar(
    outputs: torch.Tensor, labels: torch.Tensor, numberOfGestures: int = 12
):
    """
    compute accuracy for SRCnnTinyRadarNNtiny model

    Args:
        outputs (torch.Tensor): the outputs from TinyRadarNN model
        labels (torch.Tensor): labels for the data
        numberOfGestures (int): number of classes

    Returns:
        correct (float): the number of correct predictions
        total (float): the total number of predictions
    """
    return acc_tiny_radar(outputs[1], labels[1], numberOfGestures)
