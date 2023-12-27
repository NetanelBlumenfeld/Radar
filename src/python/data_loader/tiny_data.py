import os
from collections import namedtuple
from functools import partial
from multiprocessing import Pool
from typing import Optional

import cv2
import numpy as np
from data_loader.tiny_preprocess import pipeline_sr
from utils.utils_images import Normalization, normalize_img
from utils.utils_paths import ensure_path_exists

numberOfInstanceWindows = 3
lengthOfSubWindow = 32


def loadPerson(paramList, data_type: str):
    SubjectData = list()
    SubjectLabel = list()
    for gestureIdx, gestureName in enumerate(paramList.listOfGestures):
        # Create filename
        if data_type == "doppler":
            filename = (
                "p"
                + str(paramList.personIdx)
                + "/"
                + gestureName
                + "_1s_wl"
                + str(paramList.lengthOfWindow)
                + "_"
                + "doppl.npy"
            )
        elif data_type == "npy":
            filename = "p" + str(paramList.personIdx) + "/" + gestureName + "_1s.npy"

        try:
            GestureData = np.load(paramList.pathToFeatures + filename)
        except IOError:
            print("Could not open file: " + filename)
            continue
        else:
            if 0 >= GestureData.shape[0]:
                print("Skip datafile (no data): " + filename)
                continue

            SubjectData.append(GestureData)

            for idx in range(0, GestureData.shape[0]):
                GestureLabel = list()
                for jdx in range(0, GestureData.shape[1]):
                    GestureLabel.append(
                        np.identity(len(paramList.listOfGestures))[gestureIdx]
                    )
                SubjectLabel.append(np.asarray(GestureLabel))

            # Check if there is some data for this person
            if (0 >= len(SubjectData)) or (0 >= len(SubjectLabel)):
                print("No entries found for person with index 'p" + str(idx) + "'")
                return

    return np.concatenate(SubjectData, axis=0), np.asarray(SubjectLabel)


def load_data(
    pathtoDataset,
    listOfPeople,
    listofGestures,
    numberofInstanceCopies,
    lengthOfWindow,
    data_type,
) -> list:
    ParamList = namedtuple(
        "ParamList",
        "personIdx, pathToFeatures, listOfGestures, numberOfInstanceCopies, lengthOfWindow",
    )
    personList = []
    for i in listOfPeople:
        personList.append(
            ParamList(
                i,
                pathtoDataset,
                listofGestures,
                numberofInstanceCopies,
                lengthOfWindow,
            )
        )
    loadperson = partial(loadPerson, data_type=data_type)
    featureList = list(map(loadperson, personList))
    return featureList


def load_tiny_data(
    data_dir: str,
    people: list[int],
    gestures: list[str],
    row_factor: Optional[int] = None,
    col_factor: Optional[int] = None,
    up_sample: Optional[bool] = None,
    pix_norm: Optional[Normalization] = None,
    data_type: str = "doppler",
    task: str = "classification",
) -> tuple[np.ndarray, np.ndarray]:
    featureList = load_data(
        data_dir,
        people,
        gestures,
        numberOfInstanceWindows,
        lengthOfSubWindow,
        data_type,
    )
    dataX, dataY = np.zeros(0), np.empty(0)
    if task == "classification":
        dataX = np.concatenate(list(map(lambda x: x[0], featureList)), axis=0)
        dataY = np.concatenate(list(map(lambda x: x[1], featureList)), axis=0)
    elif task == "sr":
        x = list(map(lambda x: x[0], featureList))
        num_workers = os.cpu_count()
        print(f"down sampling data with {num_workers} cpus")
        with Pool(num_workers) as p:
            pipeline_sr_partial = partial(
                pipeline_sr,
                row_factor=row_factor,
                col_factor=col_factor,
                up_sample=up_sample,
                pix_norm=pix_norm,
            )
            res = list(p.map(pipeline_sr_partial, x))
            dataY = np.concatenate(list(map(lambda x: x[0], res)), axis=0)
            dataX = np.concatenate(list(map(lambda x: x[1], res)), axis=0)

    return dataX, dataY


def load_tiny_data_for_sr_off_disk(
    data_dir: str, people: list[int], gestures: list[str]
):
    pass


# def down_sample_and_save(
#     folder_path,
#     row_factor,
#     col_factor,
#     gestures,
#     people,
#     original_dim: bool,
#     save: bool = False,
# ) -> Optional[tuple[list, list]]:
#     """
#     Down samples all the .npy files in a data_npy folder by the given row and column factors and saves the down sampled
#     doppler-range map in a new folder with the same name as the original folder, but with the suffix
#     "_down_sample_row_{row_factor}_col_{col_factor}".

#     Parameters:
#     folder_path (str): The path to the folder containing the .npy files to be down sampled.
#     row_factor (int): The down sampling factor for rows.
#     col_factor (int): The down sampling factor for columns.

#     Returns:
#     None
#     """
#     npy_folder_path = folder_path + "/data_npy"
#     new_folder_path = (
#         folder_path + f"/row_{row_factor}_col_{col_factor}_d_orgdim_{original_dim}"
#     )
#     low_res, high_res = [], []
#     ensure_path_exists(new_folder_path)
#     for gdx, gestureName in enumerate(gestures):
#         for pdx, person in enumerate(people):
#             path = f"{npy_folder_path}/p{person}/{gestureName}_1s.npy"
#             print(path)
#             person_folder_path = new_folder_path + f"/p{person}"
#             file_path = person_folder_path + f"/{gestureName}_1s_wl32_doppl.npy"

#             ensure_path_exists(person_folder_path)

#             x = np.load(path)
#             x = npy_feat_reshape(x)
#             res = down_sample_data(x, row_factor, col_factor, original_dim)
#             res = doppler_maps(res, take_abs=True)
#             if save:
#                 np.save(file_path, res)
#             else:
#                 low_res.append(res)
#                 high_res.append(x)

#     return high_res, low_res
