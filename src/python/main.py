import torch as torch
from training_scripts.training_srcnn_tiny_radar import train_srcnn_tiny_radar
from training_scripts.training_tiny_radar import train_tiny_radar

if __name__ == "__main__":
    gestures = [
        "PinchIndex",
        "PinchPinky",
        "FingerSlider",
        "FingerRub",
        "SlowSwipeRL",
        "FastSwipeRL",
        "Push",
        "Pull",
        "PalmTilt",
        "Circle",
        "PalmHold",
        "NoHand",
    ]
    persons = 2
    people = list(range(1, persons, 1))

    # Dataset parameters
    numberOfTimeSteps = 5
    numberOfSensors = 2
    numberOfRangePointsPerSensor = 492
    numberOfInstanceWindows = 3
    lengthOfSubWindow = 32
    numberOfGestures = 12
    batch_size = 64
    epochs = 100

    pc = "mac"

    if pc == "4090":
        data_dir = "/mnt/netaneldata/11G/"
        output_dir = "/home/aviran/netanel/Radar/outputs/"
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
    elif pc == "mac":
        data_dir = "/Users/netanelblumenfeld/Desktop/data/11G/"
        output_dir = "/Users/netanelblumenfeld/Desktop/bgu/Msc/project/outputs/"
        device = torch.device("cpu")
    elif pc == "3080":
        data_dir = "/mnt/data/Netanel/111G/11G/"
        output_dir = "/home/aviran/Netanel/project/Radar/outputs/"
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

    print(device)
    if pc == "42090":
        train_srcnn_tiny_radar(
            gestures=gestures,
            people=people,
            output_dir=output_dir,
            experiment_name="",
            data_dir=data_dir,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
        )

    if pc == "4090":
        for data_name in ["data_feat/","_row_4_col_4_d_none_u_cubic/"]:
            data_path = data_dir + data_name
            train_tiny_radar(
                gestures=gestures,
                people=people,
                output_dir=output_dir,
                experiment_name=data_name,
                data_dir=data_path,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
            )
            break
