from dataclasses import dataclass, field
from typing import Protocol

import torch


class MetricTracker(Protocol):
    def reset(self):
        "restart the metric tracker"

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        "update the metric tracker with new data"

    @property
    def value(self) -> dict[str, float]:
        "return the value of the metric tracker"


class LossMetric:
    def __init__(self, metric_function):
        self.metric_function = metric_function
        self.name = "loss_" + metric_function.name
        self.running_total = 0
        self.values = []

    @property
    def value(self) -> dict[str, float]:
        return {"loss": sum(self.values) / self.running_total}

    def reset(self):
        self.values = []
        self.running_total = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        self.running_total += 1
        loss = self.metric_function(outputs, labels)
        self.values.append(loss.item())
        return loss


class LossMetricSRTinyRadarNN:
    def __init__(self, metric_function):
        self.metric_function = metric_function
        self.name = f"loss_{metric_function.name}"

        self.loss_srcnn_list = []
        self.loss_classifier_list = []
        self.values = []
        self.running_total = 0

    @property
    def value(self) -> dict[str, float]:
        return {
            "loss": sum(self.values) / (self.running_total + 1e-5),
            "loss_srcnn": sum(self.loss_srcnn_list) / (self.running_total + 1e-5),
            "loss_classifier": sum(self.loss_classifier_list)
            / (self.running_total + 1e-5),
        }

    def reset(self):
        self.values = []
        self.loss_classifier_list = []
        self.loss_srcnn_list = []
        self.running_total = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        self.running_total += 1
        loss, loss_classifier, loss_srcnn = self.metric_function(outputs, labels)
        self.values.append(loss.item())
        self.loss_srcnn_list.append(loss_srcnn.item())
        self.loss_classifier_list.append(loss_classifier.item())
        return loss


@dataclass
class AccuracyMetric:
    values: list[float] = field(default_factory=list)
    running_total: float = 0.0
    metric_function: callable = field(default_factory=None)

    @property
    def value(self):
        return {"acc": 100 * (sum(self.values) / self.running_total)}

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        total, correct = self.metric_function(outputs, labels)
        self.values.append(correct)
        self.running_total += total

    def reset(self):
        self.values = []
        self.running_total = 0.0
