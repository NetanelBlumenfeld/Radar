import os
from tabnanny import verbose
from typing import Optional

import torch as torch
from data_loader.tiny_radar_loader import tiny_radar_for_sr_classifier_on_disk
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
from network.models.super_resolution.srcnn import SRCnn
from network.runner import Runner
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
):
    # Training parameters
    lr = 0.001

    # paths
    tiny_radar_wights_path = (
        output_dir
        + "models/classifier/tiny_radar/data_feat/lr_0.001/time_2023-12-16_15:31:47/max_acc_model.pt"
    )
    for w_sr, w_c in zip([0], [1]):
        for n_feat1, n_feat2 in zip([32], [32]):
            for loss_type in [LossType.L1, LossType.MSE]:
                for ksize in [(3, 3), (7, 7)]:
                    for activation in ["leaky_relu", "elu"]:
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
                            "sr_classifier",  # model type
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
