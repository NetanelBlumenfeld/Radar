import torch as torch
from data_loader.tiny_radar_loader import get_tiny_radar_data_loader
from network.experiment_tracker import BaseTensorBoardTracker, SaveModel
from network.metric.accuracy import acc_tiny_radar
from network.metric.loss import LossFunctionTinyRadarNN
from network.metric.metric_tracker import AccuracyMetric, LossMetric
from network.models.classifiers.tiny_radar import TinyRadarNN
from network.runner import Runner
from utils.utils_paths import get_time_in_string


def train_tiny_radar(
    gestures: list[str],
    people: list[int],
    output_dir: str,
    experiment_name: str,
    data_dir: str,
    device: torch.device,
):
    # Dataswet parameters
    numberOfTimeSteps = 5
    numberOfSensors = 2
    numberOfRangePointsPerSensor = 492
    lengthOfSubWindow = 32
    numberOfGestures = 12

    # training parameters
    batch_size = 128
    epochs = 10
    lr = 0.001

    for lr in [0.001]:
        training_generator, val_generator = get_tiny_radar_data_loader(
            data_dir, people, gestures, batch_size
        )
        model = TinyRadarNN(
            numberOfSensors,
            numberOfRangePointsPerSensor,
            lengthOfSubWindow,
            numberOfTimeSteps,
            numberOfGestures,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
        loss_cretrion = LossFunctionTinyRadarNN(
            numberOfTimeSteps, torch.nn.CrossEntropyLoss().to(device)
        )
        loss_metric = LossMetric(metric_function=loss_cretrion)
        acc_metric = AccuracyMetric(metric_function=acc_tiny_radar)

        # paths
        data_name = data_dir.split("/")[-2]
        experiment_name = (
            f"classifier/tiny_radar/{data_name}/lr_{lr}/time_{get_time_in_string()}/"
        )
        t_board_dir = output_dir + "tensorboard/" + experiment_name
        save_model_dir = output_dir + "models/" + experiment_name

        # callbacks
        t_board = BaseTensorBoardTracker(log_dir=t_board_dir, classes_name=gestures)
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
