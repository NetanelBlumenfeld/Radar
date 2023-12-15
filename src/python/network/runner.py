import io
from typing import Any

import numpy as np
import torch
from network.experiment_tracker import (
    BaseTensorBoardTracker,
    SaveModel,
    str_acc,
    str_loss,
)
from network.metric.metric_tracker import AccuracyMetric, LossMetric
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


class Runner:
    def __init__(
        self,
        model: torch.nn.Module,
        loader_train: DataLoader,
        loader_validation: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_metric: LossMetric,
        acc_metric: AccuracyMetric,
        tboard: BaseTensorBoardTracker,
        saver: SaveModel,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.loader_validation = loader_validation
        self.acc_metric = acc_metric
        self.loss_metric = loss_metric
        self.y_true_batches: list[list[Any]] = []
        self.y_pred_batches: list[list[Any]] = []
        self.tboard = tboard
        self.saver = saver
        self.max_val_acc = 0
        self.min_val_loss = 99999999

    def reset(self):
        self.y_true_batches = []
        self.y_pred_batches = []
        self.acc_metric.reset()
        self.loss_metric.reset()

    def run(self, epochs: int):
        bar_format = "{l_bar}{bar}| [{n_fmt}/{total_fmt}]  {postfix}"
        self.tboard.writer.add_graph(
            self.model, torch.rand(5, 32, 2, 32, 492).to(self.device)
        )
        # Create a tqdm object
        pbar = tqdm(
            self.loader_train,
            total=len(self.loader_train),
            bar_format=bar_format,
            # ncols=200,
        )

        for i in range(epochs):
            pbar.set_description(f"Epoch {i}")
            self.tboard.set_mode("train")
            self.train_epoch(pbar)
            loss = self.loss_metric.value
            acc = self.acc_metric.value
            f_acc_t = str_acc(self.acc_metric.value)
            f_loss_t = str_loss(self.loss_metric.value)
            self.tboard.add_loss(loss, i)
            self.tboard.add_acc(acc, i)

            self.reset()
            self.tboard.set_mode("validation")
            self.validate_epoch()
            loss_val = self.loss_metric.value
            acc_val = self.acc_metric.value
            f_acc_v = str_acc(acc_val)
            f_loss_v = str_loss(loss_val)
            self.tboard.add_loss(self.loss_metric.value, i)
            self.tboard.add_acc(self.acc_metric.value, i)
            self.saver.save_model(self.model, acc_val["Acc"], loss_val["loss"])
            self.reset()
            f = f"Train - {f_acc_t}, {f_loss_t}  || Val - {f_acc_v}, {f_loss_v} "

            pbar.set_postfix_str(f)
            pbar.update(1)
            pbar.reset()
            # pbar.update(1)
        pbar.close()
        self.test_epoch()

    def train_epoch(self, pbar):
        self.model.train()

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
            f_acc_t = str_acc(self.acc_metric.value)
            f_loss_t = str_loss(self.loss_metric.value)
            f = f"Train - {f_acc_t}, {f_loss_t}   "
            pbar.set_postfix_str(f)

            pbar.update(1)

    def validate_epoch(self):
        self.model.eval()
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
