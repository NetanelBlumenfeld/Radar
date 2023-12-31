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
from network.metric.accuracy import acc_srcnn_tiny_radar
from network.metric.loss import LossFunctionSRTinyRadarNN, LossType
from network.metric.metric_tracker import AccuracyMetric, LossMetricSRTinyRadarNN
from network.models.classifiers.tiny_radar import TinyRadarNN
from network.models.sr_classifier.SRCnnTinyRadar import CombinedSRCNNClassifier
from network.models.super_resolution.drln import Drln
from network.models.super_resolution.srcnn import SRCnn
from network.runner import Runner
from torch.utils.data import DataLoader
from utils.utils_images import Normalization
from utils.utils_paths import get_time_in_string


def train_srcnn_tiny_radar(
    high_res_dir: str,
    low_res_dir: str,
    output_dir: str,
    gestures: list[str],
    people: list[int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    classifier_wights: Optional[str] = None,
    verbose: int = 0,
):
    # Training parameters
    lr = 0.001

    for n_feat1, n_feat2 in zip([32, 64], [32, 64]):
        for ksize in [(3, 3), (7, 7)]:
            for loss_type in [LossType.Huber, LossType.L1, LossType.MSE]:
                for activation in ["leaky_relu", "elu"]:
                    for w_sr, w_c in zip([1, 0], [1, 1]):
                        (
                            training_generator,
                            val_generator,
                            dataset_name,
                        ) = tiny_radar_for_sr_classifier_on_disk(
                            high_res_dir,
                            low_res_dir,
                            people,
                            gestures,
                            batch_size,
                            Normalization.Range_neg_1_1,
                        )
                        print(
                            f"dataset name: {dataset_name}, batch size: {batch_size}, num of train and val batches {len(training_generator)} , {len(val_generator)}"  # noqa
                        )

                        # loading models
                        tiny_radar = TinyRadarNN().to(device)
                        if classifier_wights is not None:
                            tiny_radar_wights_path = output_dir + classifier_wights
                            tiny_radar.load_state_dict(
                                torch.load(tiny_radar_wights_path)
                            )
                            for param in tiny_radar.parameters():
                                param.requires_grad = False
                        srcnn = SRCnn(
                            num_features_1=n_feat1,
                            num_features_2=n_feat2,
                            kernel_size=ksize,
                            activation=activation,
                        ).to(device)
                        model = CombinedSRCNNClassifier(srcnn, tiny_radar).to(device)

                        # models configs
                        optimizer = torch.optim.Adam(
                            model.parameters(), lr=lr, amsgrad=True
                        )
                        loss_func = LossFunctionSRTinyRadarNN(
                            loss_type_srcnn=loss_type,
                            loss_type_classifier=LossType.CrossEntropy,
                            wight_srcnn=w_sr,
                            wight_classifier=w_c,
                        )
                        loss_metric = LossMetricSRTinyRadarNN(metric_function=loss_func)
                        acc_metric = AccuracyMetric(
                            metric_function=acc_srcnn_tiny_radar
                        )

                        # paths
                        train_config = f"lr_{lr}_batch_size_{batch_size}_{loss_metric.name}_w_sr_{w_sr}_w_c_{w_c}"
                        experiment_name = os.path.join(
                            "sr_classifierfroze",  # model type
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
                            training_generator,
                            training_desc=experiment_name,
                            verbose=verbose,
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


def train_drln_tiny_radar(
    training_generator: DataLoader,
    val_generator: DataLoader,
    transform_cfg: DataTransformCfg,
    output_dir: str,
    gestures: list[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    tiny_checkpoint: Optional[str] = None,
    drln_checkpoint: Optional[str] = None,
    verbose: int = 1,
):
    # Training parameters
    lr = 0.0015
    w_sr = 1
    w_c = 1
    loss_type_sr = LossType.L1
    froze_sr = False
    froze_classifier = False

    print(
        f"dataset name: {transform_cfg}, batch size: {batch_size}, num of train and val batches {len(training_generator)} , {len(val_generator)} "
    )
    # models setup
    if tiny_checkpoint:
        classifier, _, _, _ = TinyRadarNN.load_model(
            model_dir=tiny_checkpoint,
            optimizer_class=torch.optim.Adam,
            optimizer_args=lr,
            device=device,
        )
    else:
        classifier = TinyRadarNN(numberOfGestures=len(gestures)).to(device)
    if drln_checkpoint:
        sr, _, _, _ = Drln.load_model(
            model_dir=drln_checkpoint,
            optimizer_class=torch.optim.Adam,
            optimizer_args={"lr": lr},
            device=device,
        )
    else:
        sr = Drln(2).to(device)
    if froze_sr:
        sr.freeze_weights()
    if froze_classifier:
        classifier.freeze_weights()
    model = CombinedSRCNNClassifier(
        sr, classifier, scale_factor=transform_cfg.down_sample_factor
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    loss_func = LossFunctionSRTinyRadarNN(
        loss_type_srcnn=loss_type_sr,
        loss_type_classifier=LossType.CrossEntropy,
        wight_srcnn=w_sr,
        wight_classifier=w_c,
    )
    loss_metric = LossMetricSRTinyRadarNN(metric_function=loss_func)
    acc_metric = AccuracyMetric(metric_function=acc_srcnn_tiny_radar)

    # paths
    train_config = (
        f"lr_{lr}_batch_size_{batch_size}_{loss_metric.name}_w_sr_{w_sr}_w_c_{w_c}"
    )
    if not tiny_checkpoint and drln_checkpoint:
        task_type = "sr_ck_classifier"
    elif not drln_checkpoint and tiny_checkpoint:
        task_type = "sr_classifier_ck"
    elif drln_checkpoint and tiny_checkpoint:
        task_type = "sr_ck_classifier_ck"
    else:
        task_type = "sr_classifier"
    experiment_name = os.path.join(
        task_type,  # model type
        model.model_name,  # model name
        f"{transform_cfg}",  # dataset name
        train_config,  # training configuration
        get_time_in_string(),
    )
    print(f"experiment name - {experiment_name}")
    t_board_dir = output_dir + "tensorboard/" + experiment_name
    save_model_dir = output_dir + "models/" + experiment_name

    # callbacks
    t_board = BaseTensorBoardTracker(
        log_dir=t_board_dir,
        classes_name=gestures,
        best_model_path=save_model_dir,
    )
    saver = SaveModel(save_model_dir)
    prog_bar = ProgressBar(
        training_generator,
        training_desc=experiment_name,
        verbose=verbose,
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
