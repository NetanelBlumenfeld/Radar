from dataclasses import dataclass, field

import torch


@dataclass
class AccuracyMetric:
    values: list[float] = field(default_factory=list)
    running_total: float = 0.0
    metric_function: callable = field(default_factory=None)

    @property
    def value(self):
        return 100 * (sum(self.values) / self.running_total)

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        total, correct = self.metric_function(outputs, labels)
        self.values.append(correct)
        self.running_total += total

    def reset(self):
        self.values = []
        self.running_total = 0.0


@dataclass
class LossMetric:
    values: list[float] = field(default_factory=list)
    metric_function: callable = torch.nn.CrossEntropyLoss

    @property
    def value(self):
        return sum(self.values)

    def reset(self):
        self.values = []

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        loss = self.metric_function(outputs, labels)
        self.values.append(loss.item())
        return loss


@dataclass
class LossMetricSRCnnTinyRadarNN:
    loss_srcnn_list: list[float] = field(default_factory=list)
    loss_classifier_list: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    metric_function: callable = torch.nn.CrossEntropyLoss

    @property
    def value(self):
        return sum(self.values)

    def reset(self):
        self.values = []

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        loss, loss_classifier, loss_srcnn = self.metric_function(outputs, labels)
        self.values.append(loss.item())
        self.loss_srcnn_list.append(loss_srcnn.item())
        self.loss_classifier_list.append(loss_classifier.item())
        return loss
