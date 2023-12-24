import os

import torch as torch
from data_loader.tiny_radar_loader import tiny_radar_for_sr
from network.experiment_tracker import (
    BaseTensorBoardTracker,
    CallbackHandler,
    ProgressBar,
    SaveModel,
    get_time_in_string,
)
from network.metric.loss import LossType, SimpleLoss
from network.metric.metric_tracker import AccuracyMetric, LossMetric
from network.models.super_resolution.srcnn import SRCnn
from network.runner import Runner
from utils.utils_images import Normalization


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
    lr = 0.0005
    for ksize in [(3, 3), (7, 7)]:
        for n_feat1, n_feat2 in zip([32, 64], [32, 64]):
            for activation in ["leaky_relu", "elu"]:
                for loss_type in [LossType.Huber]:
                    (
                        training_generator,
                        val_generator,
                        dataset_name,
                    ) = tiny_radar_for_sr(
                        high_res_dir,
                        low_res_dir,
                        people,
                        gestures,
                        batch_size,
                        Normalization.Range_neg_1_1,
                        test_size=0.1,
                    )
                    print(
                        f"dataset name: {dataset_name}, batch size: {batch_size}, num of train and val batches {len(training_generator)} , {len(val_generator)} "  # noqa
                    )

                    model = SRCnn(
                        num_features_1=n_feat1,
                        num_features_2=n_feat2,
                        kernel_size=ksize,
                        activation=activation,
                    ).to(device)
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=lr, amsgrad=True
                    )
                    loss_criterion = SimpleLoss(loss_function=loss_type)
                    loss_metric = LossMetric(
                        metric_function=loss_criterion, kind="loss"
                    )
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
                    prog_bar = ProgressBar(
                        training_generator, training_desc=experiment_name, verbose=0
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
