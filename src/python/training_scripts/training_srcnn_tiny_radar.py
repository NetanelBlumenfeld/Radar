import torch as torch
from data_loader.sr_tiny_radar_loader import get_sr_tiny_radar_data_loader
from network.experiment_tracker import (
    BaseTensorBoardTracker,
    CallbackHandler,
    ProgressBar,
    SaveModel,
)
from network.metric.accuracy import acc_srcnn_tiny_radar
from network.metric.loss import LossFunctionSRCnnTinyRadarNN, LossFunctionTinyRadarNN
from network.metric.metric_tracker import AccuracyMetric, LossMetricSRCnnTinyRadarNN
from network.models.classifiers.tiny_radar import TinyRadarNN
from network.models.sr_classifier.SRCnnTinyRadar import CombinedSRCNNClassifier
from network.models.super_resolution.srcnn import SRCnn
from network.runner import Runner
from utils.utils_paths import get_time_in_string


def train_srcnn_tiny_radar(
    gestures: list[str],
    people: list[int],
    output_dir: str,
    experiment_name: str,
    data_dir: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
):
    # Dataset parameters
    numberOfTimeSteps = 5
    numberOfSensors = 2
    numberOfRangePointsPerSensor = 492
    lengthOfSubWindow = 32
    numberOfGestures = 12

    # Training parameters
    lr = 0.001

    # paths
    row = 8
    col = 64
    hight_res_folder = "data_feat/"
    low_res_folder = f"data_feat_ds_row_{row}_col_{col}_d_none_u_cubic/"
    hight_res__path = data_dir + hight_res_folder
    low_res_path = data_dir + low_res_folder
    tiny_radar_wights_path = (
        output_dir
        + "models/classifier/tiny_radar/data_feat/lr_0.001/time_2023-12-16_15:31:47/max_acc_model.pt"
    )
    for w_sr, w_c in zip([0, 0.25, 0.5, 0.75, 1], [1, 0.75, 0.5, 0.25, 0]):
        for n_feat1, n_feat2 in zip([32, 64, 128, 128, 64], [32, 64, 128, 64, 32]):
            for activation in ["relu", "leaky_relu", "gelu", "tanh", "sigmoid"]:
                for ksize in [(5, 5), (3, 3), (3, 5), (5, 3)]:
                    experiment_name = f"sr_classifier/row_{row}_col_{col}_d_none_u_cubic_w_tiny/{w_c}_w_srcnn_{w_sr}/"
                    experiment_name += f"n_feat1_{n_feat1}_n_feat2_{n_feat2}_ksize_{ksize}_activation_{activation}/"
                    experiment_name += f"time_{get_time_in_string}/"
                    t_board_dir = output_dir + "tensorboard/" + experiment_name
                    save_model_dir = output_dir + "models/" + experiment_name

                    training_generator, val_generator = get_sr_tiny_radar_data_loader(
                        low_res_path, hight_res__path, people, gestures, batch_size
                    )
                    print(f"done loading data, {len(training_generator)} batches ")
                    print(experiment_name)
                    # loading models
                    tiny_radar = TinyRadarNN(
                        numberOfSensors,
                        numberOfRangePointsPerSensor,
                        lengthOfSubWindow,
                        numberOfTimeSteps,
                        numberOfGestures,
                    ).to(device)

                    tiny_radar.load_state_dict(torch.load(tiny_radar_wights_path))
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
                    tiny_loss = LossFunctionTinyRadarNN(
                        numberOfTimeSteps=numberOfTimeSteps,
                        loss_function=torch.nn.CrossEntropyLoss(),
                    )
                    srcnn_loss = LossFunctionTinyRadarNN(
                        numberOfTimeSteps=numberOfTimeSteps,
                        loss_function=torch.nn.MSELoss(),
                    )
                    loss_func = LossFunctionSRCnnTinyRadarNN(
                        loss_func_srcnn=srcnn_loss,
                        loss_func_classifier=tiny_loss,
                        wight_srcnn=w_sr,
                        wight_classifier=w_c,
                    )
                    loss_metric = LossMetricSRCnnTinyRadarNN(
                        batch_size=batch_size, metric_function=loss_func
                    )
                    acc_metric = AccuracyMetric(metric_function=acc_srcnn_tiny_radar)

                    # callbacks
                    t_board = BaseTensorBoardTracker(
                        log_dir=t_board_dir,
                        classes_name=gestures,
                        best_model_path=save_model_dir,
                    )
                    saver = SaveModel(save_model_dir)
                    prog_bar = ProgressBar(
                        training_generator, training_desc=experiment_name
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
