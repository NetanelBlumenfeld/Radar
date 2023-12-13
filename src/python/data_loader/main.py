from collections import namedtuple
from multiprocessing import Pool

import numpy as np
import torch as torch
from network.metric.loss import LossFunctionTinyRadarNN
from network.metric.metric_tracker import LossMetric
from network.models.classifiers.tiny_radar import TinyRadarNN
from network.runner import Runner
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

listGestures = [
    "Circle",
    "FastSwipeRL",
    "FingerRub",
    "FingerSlider",
    "NoHand",
    "PalmHold",
    "PalmTilt",
    "PinchIndex",
    "PinchPinky",
    "Pull",
    "Push",
    "SlowSwipeRL",
]
listPeople = range(1, 26, 1)

# Dataset parameters
numberOfTimeSteps = 5
numberOfSensors = 2
numberOfRangePointsPerSensor = 492
numberOfInstanceWindows = 3
lengthOfSubWindow = 32
numberOfGestures = 12


# Set the validation Person used for Leave-one-out cross-validation
validationPerson = 10


ParamList = namedtuple(
    "ParamList",
    "personIdx, pathToFeatures, listOfGestures, numberOfInstanceCopies, lengthOfWindow",
)


class GestureDataset(torch.utils.data.Dataset):
    def __init__(self, dataX, dataY):
        _x_train = np.concatenate([np.array(d) for d in dataX])
        print(_x_train.shape)
        self.x_train = np.transpose(_x_train, (0, 1, 4, 2, 3))

        self.tempy = np.concatenate([np.array(dy) for dy in dataY])

        self.label = np.empty((self.tempy.shape[0], self.tempy.shape[1]))
        for idx in range(self.tempy.shape[0]):
            for j in range(self.tempy.shape[1]):
                for i in range(self.tempy.shape[2]):
                    if self.tempy[idx][j][i] == 1:
                        self.label[idx][j] = i
        del self.tempy

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x_train[idx], torch.LongTensor(self.label[idx])


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
    pathtoDataset, listOfPeople, listofGestures, numberofInstanceCopies, lengthOfWindow
):
    personList = []
    for i in listOfPeople:
        personList.append(
            ParamList(
                i,
                pathToDataset,
                listofGestures,
                numberofInstanceCopies,
                lengthOfWindow,
            )
        )
    with Pool(8) as p:
        featureList = list(p.map(loadPerson, personList))
    return featureList


def setupLOOCV(dataX, dataY, personIdx):
    # Split people into train and validation set
    dataX_train = [*dataX[0:validationPerson], *dataX[validationPerson + 1 :]]
    dataY_train = [*dataY[0:validationPerson], *dataY[validationPerson + 1 :]]

    dataX_val = [dataX[validationPerson]]
    dataY_val = [dataY[validationPerson]]

    # Generate dataset from lists
    traindataset = GestureDataset(dataX_train, dataY_train)
    valdataset = GestureDataset(dataX_val, dataY_val)

    return traindataset, valdataset


if __name__ == "__main__":
    for data in ["data_feat/", "data_feat_ds_row_8_col_64_d_none_u_cubic/"]:
        # Load all previously extracted features
        path_4090 = "/mnt/netaneldata/11G"
        path_mac = "/Users/netanelblumenfeld/Desktop/data/11G/"
        pathToDataset = path_mac + data

        featureList = loadFeatures(
            pathToDataset,
            listPeople,
            listGestures,
            numberOfInstanceWindows,
            lengthOfSubWindow,
        )
        dataX = list(map(lambda x: x[0], featureList))
        dataY = list(map(lambda x: x[1], featureList))

        traindataset, valdataset = setupLOOCV(dataX, dataY, validationPerson)

        batch_size = 128
        max_epochs = 100
        # use_mps = False
        # use_mps = torch.backends.mps.is_available()
        # device = torch.device("mps" if use_mps else "cpu")
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model = TinyRadarNN(
            numberOfSensors,
            numberOfRangePointsPerSensor,
            lengthOfSubWindow,
            numberOfTimeSteps,
            numberOfGestures,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
        loss_cretrion = LossFunctionTinyRadarNN(
            numberOfTimeSteps, torch.nn.CrossEntropyLoss().to(device)
        )
        loss_metric = LossMetric(metric_function=loss_cretrion)

        training_generator = torch.utils.data.DataLoader(
            traindataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_generator = torch.utils.data.DataLoader(
            valdataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        output_dir = "/Users/netanelblumenfeld/Desktop/bgu/Msc/project/python/outputs/classifier/"
        tboard = SummaryWriter(
            log_dir=output_dir + "tensor_boards_logs/" + data,
            max_queue=2,
        )

        torch.cuda.empty_cache()

        runner = Runner(
            model,
            training_generator,
            val_generator,
            device,
            optimizer,
            loss_metric,
            tboard,
        )
        runner.run(
            2,
            output_dir + "models/" + data + "model.pt",
        )
