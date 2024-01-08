import os
from typing import Optional

import torch as torch
from config import DataTransformCfg
from network.experiment_tracker import (
    BaseTensorBoardTracker,
    CallbackHandler,
    ProgressBar,
    SaveModel,
)
from network.metric.loss import LossType, SimpleLoss
from network.metric.metric_tracker import LossMetric
from network.models.super_resolution.drln import Drln
from network.runner import Runner
from torch.utils.data import DataLoader
from utils.utils_images import Normalization
from utils.utils_paths import get_time_in_string


def train_scrnn(
    high_res_dir: str,
    low_res_dir: str,
    output_dir: str,
    gestures: list[str],
    people: list[int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    verbose: int = 0,
):
    pix_norm = Normalization.Range_0_1
    lr = 0.001
    (
        training_generator,
        val_generator,
        dataset_name,
    ) = tiny_radar_of_disk(
        high_res_dir,
        4,
        4,
        False,
        people,
        gestures,
        False,
        batch_size,
        pix_norm,
        test_size=0.1,
    )
    print(
        f"dataset name: {dataset_name}, batch size: {batch_size}, num of train and val batches {len(training_generator)} , {len(val_generator)} "  # noqa
    )
    model = Drln().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    loss_criterion = SimpleLoss(loss_function=LossType.L1)
    loss_metric = LossMetric(metric_function=loss_criterion, kind="loss")
    acc_metric = LossMetric(metric_function=loss_criterion, kind="acc")

    # paths
    train_config = f"lr_{lr}_batch_size_{batch_size}_{loss_metric.name}"
    experiment_name = os.path.join(
        "sr",  # model type
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
        log_dir=t_board_dir,
        classes_name=gestures,
        best_model_path=save_model_dir,
    )
    saver = SaveModel(save_model_dir)
    prog_bar = ProgressBar(training_generator, training_desc=experiment_name, verbose=0)
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


def train_drln(
    training_generator: DataLoader,
    val_generator: DataLoader,
    transform_cfg: DataTransformCfg,
    output_dir: str,
    gestures: list[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    verbose: int = 0,
    checkpoint: Optional[str] = None,
):
    lr = 0.0015

    print(
        f"dataset name: {transform_cfg}, batch size: {batch_size}, num of train and val batches {len(training_generator)} , {len(val_generator)} "
    )

    model = Drln(num_drln_blocks=2, num_channels=2).to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))

        print(f"loaded model from {checkpoint}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    loss_criterion = SimpleLoss(loss_function=LossType.L1)
    loss_metric = LossMetric(metric_function=loss_criterion, kind="loss")
    acc_metric = LossMetric(metric_function=loss_criterion, kind="acc")

    # paths
    train_config = f"lr_{lr}_batch_size_{batch_size}_{loss_metric.name}"
    experiment_name = os.path.join(
        "sr",  # model type
        model.model_name,  # model name
        f"{transform_cfg}",  # dataset name
        train_config,  # training configuration
        get_time_in_string(),
    )
    t_board_dir = output_dir + "tensorboard/" + experiment_name
    save_model_dir = output_dir + "models/" + experiment_name

    print(f"save dir - {save_model_dir}")
    print(f"t_board_dir - {t_board_dir}")

    # callbacks
    t_board = BaseTensorBoardTracker(
        log_dir=t_board_dir,
        classes_name=gestures,
        best_model_path=save_model_dir,
        with_cm=False,
    )
    saver = SaveModel(save_model_dir)
    prog_bar = ProgressBar(
        training_generator, training_desc=experiment_name, verbose=verbose
    )
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
