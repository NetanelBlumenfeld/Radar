from collections import namedtuple
from functools import partial

import numpy as np
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
    data_type: str = "doppler",
) -> tuple[np.ndarray, np.ndarray]:
    featureList = load_data(
        data_dir,
        people,
        gestures,
        numberOfInstanceWindows,
        lengthOfSubWindow,
        data_type,
    )
    dataX = np.concatenate(list(map(lambda x: x[0], featureList)), axis=0)
    dataY = np.concatenate(list(map(lambda x: x[1], featureList)), axis=0)
    return dataX, dataY
