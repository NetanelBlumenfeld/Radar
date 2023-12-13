import io
import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from network.metric.accuracy import acc_tiny_radar
from network.metric.metric_tracker import AccuracyMetric, LossMetric
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from python.network.experiment_tracker import BaseTensorBoardTracker


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    figure = plt.figure(figsize=(16, 16))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, f"{cm[i, j]:.2%}", horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def cm_tensor_board(true_label, pred_label, class_name):
    cm = confusion_matrix(true_label, pred_label)
    cm_figure = plot_confusion_matrix(cm, class_name)
    image = plot_to_image(cm_figure)
    return image


class Runner:
    def __init__(
        self,
        model: torch.nn.Module,
        loader_train: DataLoader,
        loader_validation: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_metric: LossMetric,
        tboard: BaseTensorBoardTracker,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.loader_validation = loader_validation
        self.acc_metric = AccuracyMetric(metric_function=acc_tiny_radar)
        self.loss_metric = loss_metric
        self.y_true_batches: list[list[Any]] = []
        self.y_pred_batches: list[list[Any]] = []
        self.tboard = tboard
        self.max_val_acc = 0
        self.min_val_loss = 99999999

    def reset(self):
        self.y_true_batches = []
        self.y_pred_batches = []
        self.acc_metric.reset()
        self.loss_metric.reset()

    def run(self, epochs: int, save_path: str):
        for i in tqdm(range(epochs)):
            self.model.train()
            self.tboard.set_mode("train")
            self.train_epoch()
            self.tboard.add_loss(self.loss_metric.value, i)
            self.tboard.add_acc(self.acc_metric.value, i)

            print(
                f"Train Stage Result:\n Loss: {self.loss_metric.value:.2f}, Acc: {self.acc_metric.value:.2f}"
            )
            self.reset()
            self.model.eval()
            self.tboard.set_mode("validation")
            self.validate_epoch()
            print(
                f"Validation Stage Result:\n Loss: {self.loss_metric.value:.2f}, Acc: {self.acc_metric.value:.2f}"
            )
            self.tboard.add_loss(self.loss_metric.value, i)
            self.tboard.add_acc(self.acc_metric.value, i)
            if self.acc_metric.value > self.max_val_acc:
                self.max_val_acc = self.acc_metric.value
                name = (
                    f"row_1_none_col_1_none_acc_max_acc_{self.acc_metric.value:.2f}.pt"
                )
                path = save_path + name
                torch.save(self.model.state_dict(), path)
                print(f"Saved model - {name}")
            if self.loss_metric.value < self.min_val_loss:
                self.min_val_loss = self.loss_metric.value
                name = f"row_1_none_col_1_none_loss_min_loss_{self.loss_metric.value:.2f}.pt"
                path = save_path + name
                torch.save(self.model.state_dict(), path)
                print(f"Saved model - {name}")
            self.reset()
        self.test_epoch()

    def train_epoch(self):
        for batch, labels in self.loader_train:
            batch, labels = self.model.reshape_to_model_output(
                batch, labels, self.device
            )

            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.loss_metric.update(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.acc_metric.update(outputs, labels)

    def validate_epoch(self):
        for batch, labels in self.loader_validation:
            # Transfer to GPU
            batch, labels = self.model.reshape_to_model_output(
                batch, labels, self.device
            )

            outputs = self.model(batch)
            _ = self.loss_metric.update(outputs, labels)
            self.acc_metric.update(outputs, labels)

    def test_epoch(self):
        preds, trues = [], []
        for batch, labels in self.loader_validation:
            # Transfer to GPU
            batch, labels = self.model.reshape_to_model_output(
                batch, labels, self.device
            )

            outputs = self.model(batch)
            outputs = outputs.cpu().detach().numpy().reshape(-1, 12)
            outputs = np.argmax(outputs, axis=1)
            preds.append(outputs)
            trues.append(labels.cpu().detach().numpy().reshape(-1))
        self.tboard.add_cm(preds, trues)
