import os

import torch as torch
from network.experiment_tracker import (
    BaseTensorBoardTracker,
    CallbackHandler,
    ProgressBar,
    SaveModel,
    get_time_in_string,
)
from network.metric.accuracy import acc_tiny_radar
from network.metric.loss import LossFunctionTinyRadarNN, LossType
from network.metric.metric_tracker import AccuracyMetric, LossMetric
from network.models.classifiers.tiny_radar import TinyRadarNN
from network.runner import Runner
from torch.utils.data import DataLoader


def train_tiny_radar(
    training_generator: DataLoader,
    val_generator: DataLoader,
    dataset_name: str,
    output_dir: str,
    gestures: list[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
):
    for lr in [0.001]:
        print(
            f"dataset name: {dataset_name}, batch size: {batch_size}, num of train and val batches {len(training_generator)} , {len(val_generator)} "
        )

        model = TinyRadarNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
        loss_criterion = LossFunctionTinyRadarNN(LossType.CrossEntropy)
        loss_metric = LossMetric(metric_function=loss_criterion, kind="loss")
        acc_metric = AccuracyMetric(metric_function=acc_tiny_radar)

        # paths
        train_config = (
            f"lr_{lr}_batch_size_{batch_size}_{loss_metric.name}_with_lowpass"
        )
        experiment_name = os.path.join(
            "classifier",  # model type
            model.model_name,  # model name
            dataset_name,  # dataset name
            train_config,  # training configuration
            get_time_in_string(),
        )
        t_board_dir = output_dir + "tensorboard/" + experiment_name
        save_model_dir = output_dir + "models/" + experiment_name

        print(f"save dir - {save_model_dir}")
        print(f"t_board_dir - {t_board_dir}")

        # callbacks
        t_board = BaseTensorBoardTracker(
            log_dir=t_board_dir, classes_name=gestures, best_model_path=save_model_dir
        )
        saver = SaveModel(save_model_dir)
        prog_bar = ProgressBar(training_generator, training_desc=experiment_name)
        callbacks = CallbackHandler([t_board, saver, prog_bar])
        torch.cuda.empty_cache()

        runner = Runner(
            model,
            training_generator,
            val_generator,
            device,
            optimizer,
            loss_metric,
            acc_metric,
            callbacks,
        )
        runner.run(epochs)
