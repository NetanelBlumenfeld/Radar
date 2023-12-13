import os

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.tensorboard import SummaryWriter


def ensure_path_exists(path):
    """
    Checks if a given path exists, and if not, creates it.

    Parameters:
    path (str): The path to be checked and potentially created.

    Returns:
    None
    """
    if not os.path.exists(path):
        os.makedirs(path)


class BaseTensorBoardTracker:
    def __init__(self, log_dir: str, classes_name: list[str]):
        ensure_path_exists(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.classes_name = classes_name
        self.mode = "train"

    def set_mode(self, mode: str):
        self.mode = mode

    def add_loss(self, loss, epoch):
        self.writer.add_scalar("loss/" + self.mode, loss, epoch)

    def add_acc(self, acc, epoch):
        self.writer.add_scalar("acc/" + self.mode, acc, epoch)

    def add_cm(self, cm, trues, preds):
        cm = confusion_matrix(np.concatenate(trues), np.concatenate(preds))
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        # Create the ConfusionMatrixDisplay instance
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=cm_normalized,
            display_labels=self.classes_name,
        )

        # Plot with percentages
        cm_display.plot(cmap="Blues", values_format=".2%")
        self.writer.add_figure("confution_matrix/cm", cm_display.figure_, 0)


class TensorBoardTrackerSRCnnTinyRadarNN(BaseTensorBoardTracker):
    def __init__(self, log_dir: str):
        super().__init__(log_dir=log_dir)

    def add_loss(self, loss, epoch):
        total_loss = loss[0]
        srcnn_loss = loss[1]
        classifier_loss = loss[2]
        self.writer.add_scalar("total_loss/" + self.mode, total_loss, epoch)
        self.writer.add_scalar("srcnn_loss/" + self.mode, srcnn_loss, epoch)
        self.writer.add_scalar("classifier_loss/" + self.mode, classifier_loss, epoch)
