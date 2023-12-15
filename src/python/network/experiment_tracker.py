import datetime
import os

import numpy as np
import torch
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


def get_time_in_string():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H:%M:%S")


class BaseTensorBoardTracker:
    def __init__(self, log_dir: str, classes_name: list[str]):
        ensure_path_exists(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.classes_name = classes_name
        self.mode = "train"

    def set_mode(self, mode: str):
        self.mode = mode

    def add_loss(self, loss, epoch):
        self.writer.add_scalar("loss/" + self.mode, loss["loss"], epoch)

    def add_acc(self, acc, epoch):
        self.writer.add_scalar("acc/" + self.mode, acc["Acc"], epoch)

    def add_cm(self, trues, preds):
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
    def __init__(self, log_dir: str, classes_name: list[str]):
        super().__init__(log_dir=log_dir, classes_name=classes_name)

    def add_loss(self, loss, epoch):
        self.writer.add_scalar("total_loss/" + self.mode, loss["loss"], epoch)
        self.writer.add_scalar("srcnn_loss/" + self.mode, loss["loss_srcnn"], epoch)
        self.writer.add_scalar(
            "classifier_loss/" + self.mode["loss_classifier"], loss, epoch
        )


class SaveModel:
    def __init__(self, save_path: str):
        ensure_path_exists(save_path)
        self.save_path = save_path
        self.max_val_acc = 0
        self.min_val_loss = 99999999

    def save_model(self, model, acc, loss):
        if acc > self.max_val_acc:
            self.max_val_acc = acc
            self.min_val_loss = loss
            name = f"max_acc_{acc:.2f}_loss_{loss}.pt"
            path = self.save_path + name
            torch.save(model.state_dict(), path)
            print(f"\nSaved model - {name}")
        if loss < self.min_val_loss:
            self.min_val_loss = loss
            self.max_val_acc = acc
            name = f"acc_{acc}_min_loss_{loss:.2f}.pt"
            path = self.save_path + name
            torch.save(model.state_dict(), path)
            print(f"\nSaved model - {name}")


def str_acc(acc):
    return f"Acc: {acc['Acc']:.2f}"


def str_loss(losses: dict):
    return ", ".join([f"{k}: {v:.3f}" for k, v in losses.items()])


if __name__ == "__main__":
    import time

    from tqdm import tqdm

    # Sample iterable
    iterable = range(100)

    # Custom format:
    # - '{l_bar}': left part of the bar (includes the percentage and bar itself)
    # - '{bar}': the progress bar
    # - '{r_bar}': right part of the bar (includes the remaining time and iteration info)
    # - '{postfix}': place for your custom data
    bar_format = "{l_bar}{bar}| [{n_fmt}/{total_fmt}] {remaining}, {postfix}"

    # Initialize tqdm with the custom bar format
    with tqdm(iterable, bar_format=bar_format, ncols=200) as pbar:
        for item in pbar:
            # Update the postfix data with whatever dynamic information you want to display
            pbar.set_postfix_str(f"YourData={item}")
            pbar.set_description(f"Processing {item*2}")
            pbar.set_description_str(f"Processing2 {item*2}")
            # Simulate some work
            time.sleep(0.1)
