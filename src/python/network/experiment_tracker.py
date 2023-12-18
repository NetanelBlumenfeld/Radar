import datetime
import os
from typing import Optional, Protocol

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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


class CallbackProtocol(Protocol):
    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        ...

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        ...

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        ...

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        ...

    def on_batch_begin(
        self, batch: Optional[int] = None, logs: Optional[dict] = None
    ) -> None:
        ...

    def on_batch_end(
        self, batch: Optional[int] = None, logs: Optional[dict] = None
    ) -> None:
        ...

    def on_eval_begin(self, logs: Optional[dict] = None) -> None:
        ...

    def on_eval_end(self, logs: Optional[dict] = None) -> None:
        ...


class CallbackHandler(CallbackProtocol):
    def __init__(self, callbacks: Optional[list[CallbackProtocol]] = None):
        self.callbacks = callbacks if callbacks is not None else []

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        for callback in self.callbacks:
            if hasattr(callback, "on_train_begin"):
                callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[dict] = None):
        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, "on_epoch_begin"):
                callback.on_epoch_begin(epoch, **kwargs)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        for callback in self.callbacks:
            if hasattr(callback, "on_epoch_end"):
                callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: Optional[int] = None, logs: Optional[dict] = None):
        for callback in self.callbacks:
            if hasattr(callback, "on_batch_begin"):
                callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: Optional[int] = None, logs: Optional[dict] = None):
        for callback in self.callbacks:
            if hasattr(callback, "on_batch_end"):
                callback.on_batch_end(batch, logs)

    def on_eval_begin(self, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, "on_eval_begin"):
                callback.on_eval_begin(**kwargs)

    def on_eval_end(self, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, "on_eval_end"):
                callback.on_eval_end(**kwargs)


class BaseTensorBoardTracker(CallbackProtocol):
    def __init__(self, log_dir: str, classes_name: list[str], best_model_path: str):
        ensure_path_exists(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.classes_name = classes_name
        self.best_model_path = best_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _add_cm(self, trues, preds, title: str):
        cm = confusion_matrix(np.concatenate(trues), np.concatenate(preds))
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        # Create the ConfusionMatrixDisplay instance
        fig, ax = plt.subplots(figsize=(20, 20))
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=cm_normalized,
            display_labels=self.classes_name,
        )

        # Plot with percentages
        cm_display.plot(cmap="Blues", values_format=".2%", ax=ax)
        self.writer.add_figure(f"confusion_matrix/{title}", cm_display.figure_, 0)

    # def on_train_begin(self, logs: Optional[dict] = None) -> None:
    # model = logs["model"]
    # self.writer.add_graph(model, torch.rand(5, 32, 2, 32, 492)).to(self.device)

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        """loading the best model and calculate the confusion matrix"""

        # TODO - move to another place
        def _get_preds_for_best_models(model, loader):
            preds, trues = [], []
            for batch, labels in loader:
                batch, labels = model.reshape_to_model_output(
                    batch, labels, self.device
                )

                outputs = model(batch)
                pred_labels = outputs[1].cpu().detach().numpy().reshape(-1, 12)
                # pred_labels = outputs.cpu().detach().numpy().reshape(-1, 12)

                pred_labels = np.argmax(pred_labels, axis=1)
                preds.append(pred_labels)
                trues.append(labels[1].cpu().detach().numpy().reshape(-1))
                # trues.append(labels.cpu().detach().numpy().reshape(-1))

            return preds, trues

        model = logs["model"].to(self.device)
        models = ["max_acc_model.pt", "min_loss_model.pt"]
        data_loader = logs["data_loader"]
        for model_name in models:
            model.load_state_dict(torch.load(self.best_model_path + model_name))
            preds, trues = _get_preds_for_best_models(model, data_loader)
            self._add_cm(preds, trues, model_name.split(".")[0])
        self.writer.close()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        metrics = logs["metrics"]
        for data in ["train", "val"]:
            for metric_name, metric_value in metrics[data].items():
                self.writer.add_scalar(f"{data}/{metric_name}", metric_value, epoch)


class SaveModel(CallbackProtocol):
    def __init__(self, save_path: str):
        ensure_path_exists(save_path)
        self.save_path = save_path
        self.max_val_acc = 0
        self.min_val_loss = 99999999

    def _save(self, model, path):
        torch.save(model.state_dict(), path)
        print(f"\nSaved model at - {path}")

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        loss = logs["metrics"]["val"]["loss"]
        acc = logs["metrics"]["val"]["acc"]
        model = logs["model"]
        if acc > self.max_val_acc:
            self.max_val_acc = acc
            path = self.save_path + "max_acc_model.pt"
            self._save(model, path)
        if loss < self.min_val_loss:
            self.min_val_loss = loss
            path = self.save_path + "min_loss_model.pt"
            self._save(model, path)


class ProgressBar(CallbackProtocol):
    def __init__(self, loader_train, training_desc: Optional[str] = None):
        self.loader_train = loader_train
        self.training_desc = training_desc
        self.out_val = ""
        self.out_train = ""

    def _print_metrics(self, metrics: dict[str, dict]) -> str:
        res = ""
        for data in ["train", "val"]:
            for metric_name, metric_value in metrics[data].items():
                res += f"{metric_name}: {metric_value:.2f} "
        return res

    def _update_postfix_str(self):
        self.pbar.set_postfix_str(f"{self.out_train} {self.out_val}")

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        print(f"Start Training\n {self.training_desc}")
        bar_format = "{l_bar}{bar}| [{n_fmt}/{total_fmt}]  {postfix}"
        self.pbar = tqdm(
            self.loader_train,
            total=len(self.loader_train),
            bar_format=bar_format,
            ncols=200,
        )

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        self.pbar.close()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        self.pbar.reset()
        self.out_val = ""
        self.out_train = ""
        print("\n")

        self.pbar.set_description(f"Epoch {epoch}")

    def on_batch_end(
        self, batch: Optional[int] = None, logs: Optional[dict] = None
    ) -> None:
        metrics = logs["metrics"]
        self.out_train = f"Train - {self._print_metrics(metrics)}"
        self._update_postfix_str()
        self.pbar.update(1)

    def on_eval_end(self, logs: Optional[dict] = None) -> None:
        metrics = logs["metrics"]
        self.out_val = f"Val - {self._print_metrics(metrics)}"
        self._update_postfix_str()


class TensorBoardTrackerSRCnnTinyRadarNN(BaseTensorBoardTracker):
    def __init__(self, log_dir: str, classes_name: list[str]):
        super().__init__(log_dir=log_dir, classes_name=classes_name)

    def add_loss(self, loss, epoch):
        self.writer.add_scalar("total_loss/" + self.mode, loss["loss"], epoch)
        self.writer.add_scalar("srcnn_loss/" + self.mode, loss["loss_srcnn"], epoch)
        self.writer.add_scalar(
            "classifier_loss/" + self.mode["loss_classifier"], loss, epoch
        )


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
            pbar.set_postfix_str(f"YourData={item}", refresh=False)
            # pbar.set_description(f"Processing {item*2}")
            # pbar.set_description_str(f"Processing2 {item*2}")
            # Simulate some work
            time.sleep(0.1)
