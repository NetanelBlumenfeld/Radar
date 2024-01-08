import os
from functools import partial

import numpy as np
import torch as torch
from config import DataCfg, DataTransformCfg, TrainCfg
from data_loader.tiny_loader import down_sample_data_sr
from data_loader.tiny_pipeline import (
    classifier_pipeline,
    sr_classifier_3080_pipeline,
    sr_classifier_4090_pipeline,
    sr_time_4090_pipeline,
)
from data_loader.tiny_radar_loader import get_tiny_dataloader
from training_scripts.training_classifier import train_tiny_radar
from training_scripts.training_sr import train_drln
from training_scripts.training_sr_classifier import train_drln_tiny_radar
from utils.utils_images import Normalization, normalize_img


def process_data_time(x: np.ndarray) -> np.ndarray:
    # TODO: adding this as preprocess pipeline
    assert len(x.shape) == 2
    assert x.dtype == np.complex64
    x = np.array([np.real(x), np.imag(x)])
    x[0] = normalize_img(x[0], Normalization.Range_0_1)
    x[1] = normalize_img(x[1], Normalization.Range_0_1)
    return x


def get_pc_cgf(pc: str) -> tuple[str, str, torch.device]:
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
        output_dir = "/home/aviran/netanel/project/Radar/outputs/"
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
    else:
        raise ValueError("pc must be 4090, mac or 3080")
    return data_dir, output_dir, device


if __name__ == "__main__":
    pc = "4090"
    task = "sr"
    data_dir, output_dir, device = get_pc_cgf(pc)

    # data config
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
    people = 26

    # experiment config
    verbose = 0

    # pc config
    batch_size = 16
    train_cfg = TrainCfg(epochs=300)

    print(
        f"Running experiment with epochs: {train_cfg.epochs}, batch_size: {batch_size} ,people: {people}, gestures: {gestures}"
    )

    print(device)
    if (pc == "3080" or pc == "1") and task == "sr_classifier":
        data_transform_cfg = DataTransformCfg(
            down_sample_factor=4,
            back_to_original_dim=False,
            pix_norm=Normalization.Range_0_1,
        )
        ds_func = partial(
            down_sample_data_sr,
            row_factor=data_transform_cfg.down_sample_factor,
            col_factor=data_transform_cfg.down_sample_factor,
            original_dim=data_transform_cfg.back_to_original_dim,
        )
        norm_func = partial(normalize_img, pix_norm=data_transform_cfg.pix_norm)
        load_data_func = partial(
            sr_classifier_3080_pipeline, norm_func=norm_func, down_sample_func=ds_func
        )

        data_cfg = DataCfg(
            task="sr_classifier",
            gestures=gestures,
            people=people,
            data_type="npy",
            transform_pipeline=load_data_func,
        )
        train, val = get_tiny_dataloader(
            data_dir=data_dir,
            data_cfg=data_cfg,
            pc="3080",
            test_size=0.1,
            batch_size=batch_size,
        )

        train_drln_tiny_radar(
            training_generator=train,
            val_generator=val,
            transform_cfg=data_transform_cfg,
            output_dir=output_dir,
            gestures=gestures,
            device=device,
            epochs=20,
            batch_size=batch_size,
        )

    if (pc == "4090" or pc == "mac") and task == "sr_classifier":
        classifier_ck = os.path.join(
            output_dir,
            "models/classifier/TinyRadar/original/lr_0.001_batch_size_128_loss_CrossEntropy_with_lowpass/2024-01-06_18:32:54max_acc_model.pt",
        )
        drln_ck = os.path.join(
            output_dir,
            "models/sr/Drln_2/11G_normalization.range_0_1/lr_0.0015_batch_size_128_loss_L1/2023-12-31_12:10:32min_loss_model.pt",
        )
        data_transform_cfg = DataTransformCfg(
            down_sample_factor=4,
            back_to_original_dim=False,
            pix_norm=Normalization.Range_0_1,
        )
        ds_func = partial(
            down_sample_data_sr,
            row_factor=data_transform_cfg.down_sample_factor,
            col_factor=data_transform_cfg.down_sample_factor,
            original_dim=data_transform_cfg.back_to_original_dim,
        )

        norm_func = partial(normalize_img, pix_norm=data_transform_cfg.pix_norm)

        load_data_func = partial(
            sr_classifier_4090_pipeline,
            norm_func=norm_func,
            down_sample_func=ds_func,
            gestures=gestures,
        )

        # data_dir = "/Users/netanelblumenfeld/Desktop/data/11G/"
        data_cfg = DataCfg(
            task="sr_classifier",
            gestures=gestures,
            people=people,
            data_type="npy",
            transform_pipeline=load_data_func,
        )
        print(batch_size)
        train, val = get_tiny_dataloader(
            data_dir=data_dir,
            data_cfg=data_cfg,
            pc="4090",
            test_size=0.1,
            batch_size=batch_size,
        )
        for x, y in train:
            print(x.shape)
            print(y[0].shape)
            print(y[1].shape)
            break

        train_drln_tiny_radar(
            training_generator=train,
            val_generator=val,
            transform_cfg=data_transform_cfg,
            output_dir=output_dir,
            gestures=gestures,
            device=device,
            epochs=150,
            batch_size=batch_size,
            drln_checkpoint=drln_ck,
        )
    if (pc == "4090" or pc == "mac") and task == "sr":
        ds_func = partial(
            down_sample_data_sr, row_factor=4, col_factor=4, original_dim=False
        )
        norm_func = partial(normalize_img, pix_norm=Normalization.Range_0_1)
        load_data_func = partial(
            sr_time_4090_pipeline, norm_func=norm_func, down_sample_func=ds_func
        )
        data_cfg = DataCfg(
            task="sr",
            gestures=gestures,
            people=people,
            data_type="npy",
            transform_pipeline=load_data_func,
        )
        train, val = get_tiny_dataloader(
            data_dir=data_dir, data_cfg=data_cfg, pc="4090", test_size=0.1, batch_size=8
        )
        data_transform_cfg = DataTransformCfg(
            down_sample_factor=4,
            back_to_original_dim=False,
            pix_norm=Normalization.Range_0_1,
        )
        train_drln(
            training_generator=train,
            val_generator=val,
            device=device,
            transform_cfg=data_transform_cfg,
            output_dir=output_dir,
            gestures=gestures,
            epochs=2,
            batch_size=batch_size,
        )

    if pc == "1":
        checkpoint = (
            output_dir
            + "models/sr/Drln_2/11G_normalization.range_0_1/lr_0.0005_batch_size_128_loss_L1/2023-12-27_13:39:19min_loss_model.pt"
        )
        train_drln(
            dir=data_dir,
            pc=pc,
            output_dir=output_dir,
            gestures=gestures,
            people=people,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            checkpoint=checkpoint,
        )

    if pc == "1":
        high_res_dir = data_dir + "data_npy/"
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
    if pc == "4090" and task == "classifier":
        # classifier
        norm_func = partial(normalize_img, pix_norm=Normalization.Range_0_1)
        load_data_func = partial(classifier_pipeline, norm_func=norm_func)
        data_dir = "/Users/netanelblumenfeld/Desktop/data/11G/"
        data_cfg = DataCfg(
            task="classifier",
            gestures=gestures,
            people=people,
            data_type="npy",
            transform_pipeline=load_data_func,
        )
        train, val = get_tiny_dataloader(
            data_dir=data_dir,
            data_cfg=data_cfg,
            pc="4090",
            test_size=0.1,
            batch_size=batch_size,
        )
        name = "original"
        train_tiny_radar(
            training_generator=train,
            val_generator=val,
            dataset_name=name,
            output_dir=output_dir,
            gestures=gestures,
            device=device,
            epochs=100,
            batch_size=batch_size,
        )
