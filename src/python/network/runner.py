from typing import Any

import torch
import torch.optim.lr_scheduler as lr_scheduler
from network.experiment_tracker import CallbackHandler
from network.metric.metric_tracker import MetricTracker
from torch.utils.data.dataloader import DataLoader


class Runner:
    def __init__(
        self,
        model: torch.nn.Module,
        loader_train: DataLoader,
        loader_validation: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_metric: MetricTracker,
        acc_metric: MetricTracker,
        callbacks: CallbackHandler,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.loader_validation = loader_validation
        self.acc_metric = acc_metric
        self.loss_metric = loss_metric
        self.callbacks = callbacks
        self.lr_s = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    def reset(self):
        self.acc_metric.reset()
        self.loss_metric.reset()

    def run(self, epochs: int):
        logs = {
            "model": self.model,
            "metrics": {"train": None, "val": None},
            "data_loader": None,
        }
        self.callbacks.on_train_begin(logs)
        for i in range(epochs):
            self.callbacks.on_epoch_begin(i)
            logs["metrics"]["train"] = self.train()
            self.reset()
            logs["metrics"]["val"] = self.validate()
            self.reset()
            self.callbacks.on_epoch_end(i, logs)
            self.lr_s.step()
            print(f"epoch {i} lr: {self.optimizer.param_groups[0]['lr']}")
        logs["data_loader"] = self.loader_validation
        self.callbacks.on_train_end(logs)

    def train(self) -> dict[str, Any]:
        self.model.train()
        logs = {"model": self.model, "metrics": {"train": {}, "val": {}}}

        for batch, labels in self.loader_train:
            batch, labels = self.model.reshape_to_model_output(
                batch, labels, self.device
            )
            batch = batch
            labels[0] = labels[0]
            self.callbacks.on_batch_begin(logs=logs)
            logs["metrics"]["train"] = self.train_batch(batch, labels)
            self.callbacks.on_batch_end(logs=logs)
        return logs["metrics"]["train"]

    def validate(self):
        self.model.eval()
        logs = {"model": self.model, "metrics": {"train": {}, "val": {}}}
        for batch, labels in self.loader_validation:
            batch, labels = self.model.reshape_to_model_output(
                batch, labels, self.device
            )
            batch = batch
            labels[0] = labels[0]
            self.callbacks.on_eval_begin()
            logs["metrics"]["val"] = self.validate_batch(batch, labels)
            self.callbacks.on_eval_end(logs=logs)
        return logs["metrics"]["val"]

    def train_batch(self, batch, labels) -> dict[str, Any]:
        self.optimizer.zero_grad()
        outputs = self.model(batch)
        loss = self.loss_metric.update(outputs, labels)
        loss.backward()  # type: ignore
        self.optimizer.step()
        self.acc_metric.update(outputs, labels)
        return self.loss_metric.value | self.acc_metric.value

    def validate_batch(self, batch, labels) -> dict[str, Any]:
        outputs = self.model(batch)
        _ = self.loss_metric.update(outputs, labels)
        self.acc_metric.update(outputs, labels)
        return self.loss_metric.value | self.acc_metric.value
