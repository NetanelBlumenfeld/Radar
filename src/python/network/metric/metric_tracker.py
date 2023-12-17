from dataclasses import dataclass, field

import torch


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


@dataclass
class LossMetric:
    values: list[float] = field(default_factory=list)
    metric_function: callable = torch.nn.CrossEntropyLoss()
    running_total: int = 0

    @property
    def value(self):
        return {"loss": sum(self.values) / self.running_total}

    def reset(self):
        self.values = []
        self.running_total = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        self.running_total += 1
        loss = self.metric_function(outputs, labels)
        self.values.append(loss.item())
        return loss


@dataclass
class LossMetricSRCnnTinyRadarNN:
    batch_size: int
    loss_srcnn_list: list[float] = field(default_factory=list)
    loss_classifier_list: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    metric_function: callable = field(default_factory=callable)
    running_total: int = 0

    @property
    def value(self):
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
