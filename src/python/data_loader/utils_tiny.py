from collections import namedtuple

import cv2
import numpy as np
from utils.utils_images import Normalization, normalize_img

numberOfInstanceWindows = 3
lengthOfSubWindow = 32


def loadPerson(paramList):
    SubjectData = list()
    SubjectLabel = list()
    print(f"Doing {paramList.personIdx}")
    for gestureIdx, gestureName in enumerate(paramList.listOfGestures):
        # Create filename
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

        # Load data gesture data from disk
#        print(paramList.pathToFeatures + filename)
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


def loadFeatures(
    pathtoDataset,
    listOfPeople,
    listofGestures,
    numberofInstanceCopies,
    lengthOfWindow,
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
    featureList = list(map(loadPerson, personList))
    return featureList


def load_tiny_doppler_maps(
    data_dir: str, pix_norm: Normalization, people: list[int], gestures: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    featureList = loadFeatures(
        data_dir,
        people,
        gestures,
        numberOfInstanceWindows,
        lengthOfSubWindow,
    )
    dataX = np.concatenate(list(map(lambda x: x[0], featureList)), axis=0)
    dataY = np.concatenate(list(map(lambda x: x[1], featureList)), axis=0)

    if pix_norm != Normalization.NONE:
        for i in range(dataX.shape[0]):
            for j in range(dataX.shape[1]):
                for k in range(dataX.shape[4]):
                    dataX[i, j, :, :, k] = normalize_img(dataX[i, j, :, :, k], pix_norm)
    return dataX, dataY


def down_sample_doppler_maps(
    data: np.ndarray, row_factor: int, col_factor: int, original_dim: bool = False
) -> np.ndarray:
    def _down_sample(img: np.ndarray, row_factor: int, col_factor: int) -> np.ndarray:
        return img[::row_factor, ::col_factor]

    def _up_scale(img: np.ndarray, dim_up: tuple[int, int]) -> np.ndarray:
        real_img = np.real(img)
        imag_img = np.imag(img)
        data_real_up = cv2.resize(real_img, dim_up, interpolation=cv2.INTER_CUBIC)
        data_imag_up = cv2.resize(imag_img, dim_up, interpolation=cv2.INTER_CUBIC)
        return data_real_up + 1j * data_imag_up

    if original_dim:
        res = np.empty_like(data)
        org_dim = (data.shape[2], data.shape[3])
    else:
        res = np.empty(
            (
                data.shape[0],
                data.shape[1],
                data.shape[2] // row_factor,
                data.shape[3] // col_factor,
                data.shape[4],
            ),
            dtype=np.complex64,
        )
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[4]):
                img = _down_sample(data[i, j, :, :, k], row_factor, col_factor)
                if original_dim:
                    img = _up_scale(img, org_dim)

                res[i, j, :, :, k] = img
    return data
