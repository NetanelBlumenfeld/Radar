import torch as torch
from data_loader.sr_tiny_radar_loader import get_sr_tiny_radar_data_loader
from network.experiment_tracker import SaveModel, TensorBoardTrackerSRCnnTinyRadarNN
from network.metric.accuracy import acc_srcnn_tiny_radar
from network.metric.loss import LossFunctionSRCnnTinyRadarNN, LossFunctionTinyRadarNN
from network.metric.metric_tracker import AccuracyMetric, LossMetricSRCnnTinyRadarNN
from network.models.classifiers.tiny_radar import TinyRadarNN
from network.models.sr_classifier.SRCnnTinyRadar import CombinedSRCNNClassifier
from network.models.super_resolution.srcnn import SRCnn
from network.runner import Runner

from python.utils.utils_paths import get_time_in_string


def train(
    gestures: list[str],
    people: list[int],
    output_dir: str,
    experiment_name: str,
    data_dir: str,
    device: torch.device,
):
    # Dataset parameters
    numberOfTimeSteps = 5
    numberOfSensors = 2
    numberOfRangePointsPerSensor = 492
    numberOfInstanceWindows = 3
    lengthOfSubWindow = 32
    numberOfGestures = 12

    # Training parameters
    classifier_wights = 0.5
    srcnn_wights = 0.5
    batch_size = 128
    epochs = 85
    lr = 0.001

    # paths
    row = 8
    col = 64
    hight_res_folder = "data_feat/"
    low_res_folder = f"data_feat_ds_row_{row}_col_{col}_d_none_u_cubic/"
    hight_res__path = data_dir + hight_res_folder
    low_res_path = data_dir + low_res_folder

    tiny_radar_wights_path = output_dir + "saved_models/data_feat/" + "MODELPATH"
    experiment_name = f"sr_classifier/row_{row}_col_{col}_d_none_u_cubic_w_tiny_{classifier_wights}_w_srcnn_{srcnn_wights}/time_{get_time_in_string}/"

    t_board_dir = output_dir + "tensor_boards_logs/" + experiment_name
    save_model_dir = output_dir + "models/" + experiment_name

    training_generator, val_generator = get_sr_tiny_radar_data_loader(
        low_res_path, hight_res__path, people, gestures, batch_size
    )
    print(f"done loading data, {len(training_generator)} batches ")
    tiny_radar = TinyRadarNN(
        numberOfSensors,
        numberOfRangePointsPerSensor,
        lengthOfSubWindow,
        numberOfTimeSteps,
        numberOfGestures,
    ).to(device)

    tiny_radar.load_state_dict(
        torch.load(tiny_radar_wights_path, map_location=torch.device("cpu"))
    )
    for param in tiny_radar.parameters():
        param.requires_grad = False

    srcnn = SRCnn().to(device)

    model = CombinedSRCNNClassifier(srcnn, tiny_radar).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    tiny_loss = LossFunctionTinyRadarNN(
        numberOfTimeSteps=numberOfTimeSteps, loss_function=torch.nn.CrossEntropyLoss()
    )
    srcnn_loss = torch.nn.MSELoss()
    loss_func = LossFunctionSRCnnTinyRadarNN(
        numberOfTimeSteps=numberOfTimeSteps,
        loss_func_srcnn=srcnn_loss,
        loss_func_classifier=tiny_loss,
        wight_srcnn=srcnn_wights,
        wight_classifier=classifier_wights,
    )
    loss_metric = LossMetricSRCnnTinyRadarNN(
        batch_size=batch_size, metric_function=loss_func
    )
    acc_metric = AccuracyMetric(metric_function=acc_srcnn_tiny_radar)

    # callbacks
    t_board = TensorBoardTrackerSRCnnTinyRadarNN(
        log_dir=t_board_dir, classes_name=gestures
    )
    saver = SaveModel(save_model_dir)
    torch.cuda.empty_cache()

    runner = Runner(
        model,
        training_generator,
        val_generator,
        device,
        optimizer,
        loss_metric,
        acc_metric,
        t_board,
        saver,
    )
    runner.run(epochs)
