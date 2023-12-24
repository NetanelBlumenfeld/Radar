import torch as torch
from training_scripts.training_classifier import train_tiny_radar
from training_scripts.training_sr import train_scrnn
from training_scripts.training_sr_classifier import train_srcnn_tiny_radar

# from training_scripts.training_srcnn_tiny_radar import train_srcnn_tiny_radar


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
    persons = 26
    people = list(range(1, persons, 1))

    batch_size = 64
    epochs = 130

    pc = "4090"
    verbose = 1
    output_dir, data_dir, device = "", "", None
    device = torch.device("cpu")

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
    if pc == "40903":
        high_res_dir = data_dir + "data_feat/"
        low_res_dir = data_dir + "_row_4_col_4_d_none_u_cubic/"
        classifier_wights = "models/classifier/TinyRadar/data_feat_normalization.range_neg_1_1/lr_0.001_batch_size_64_loss_CrossEntropy/2023-12-24_10:18:18max_acc_model.pt"
        train_srcnn_tiny_radar(
            high_res_dir=high_res_dir,
            low_res_dir=low_res_dir,
            output_dir=output_dir,
            gestures=gestures,
            people=people,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            classifier_wights=classifier_wights,
            verbose=verbose,
        )

    # if pc == "3080":
    #     for data_name in ["data_feat/"]:
    #         data_path = data_dir + data_name
    #         train_tiny_radar(
    #             data_dir=data_path,
    #             output_dir=output_dir,
    #             gestures=gestures,
    #             people=people,
    #             device=device,
    #             epochs=epochs,
    #             batch_size=batch_size,
    #         )

    if pc == "4090":
        high_res_dir = data_dir + "data_feat/"
        low_res_dir = data_dir + "_row_4_col_4_d_none_u_cubic/"
        train_scrnn(
            high_res_dir=high_res_dir,
            low_res_dir=low_res_dir,
            output_dir=output_dir,
            gestures=gestures,
            people=people,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
