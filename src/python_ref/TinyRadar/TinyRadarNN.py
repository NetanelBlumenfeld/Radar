# ----------------------------------------------------------------------
#
# File: TinyRadarNN.py
#
# Last edited: 09.11.2020
#
# Copyright (C) 2020, ETH Zurich and University of Bologna.
#
# Author: Jonas Erb & Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch as torch
import numpy as np
from multiprocessing import Pool
from collections import namedtuple
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

pathToDataset = "./11G/"

# Set the validation Person used for Leave-one-out cross-validation
validationPerson = 10

# Used for early stopping -> Save model with highest validation accuracy
max_val_acc = 0

ParamList = namedtuple(
    "ParamList",
    "personIdx, pathToFeatures, listOfGestures, numberOfInstanceCopies, lengthOfWindow",
)


class GestureDataset(torch.utils.data.Dataset):
    def __init__(self, dataX, dataY):
        _x_train = np.concatenate(np.asarray(dataX))
        print(_x_train.shape)
        self.x_train = np.transpose(_x_train, (0, 1, 4, 2, 3))

        self.tempy = np.concatenate(np.asarray(dataY))

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


class CausalConv1D(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1D, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1D, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


class cust_TCNLayer(CausalConv1D):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(cust_TCNLayer, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(cust_TCNLayer, self).forward(input)
        result = torch.nn.functional.relu(result)
        return result + input


class Model(torch.nn.Module):
    def __init__(
        self,
        numberOfSensors,
        numberOfRangePointsPerSensor,
        lengthOfWindow,
        numberOfTimeSteps,
        numberOfGestures,
    ):
        # Parameters that need to be consistent with the dataset
        super(Model, self).__init__()
        self.lWindow = lengthOfWindow
        self.nRangePoints = numberOfRangePointsPerSensor
        self.nSensors = numberOfSensors
        self.nTimeSteps = numberOfTimeSteps
        self.nGestures = numberOfGestures

        self.CNN = torch.nn.Sequential(*self.CreateCNN())
        self.TCN = torch.nn.Sequential(*self.CreateTCN())
        self.Classifier = torch.nn.Sequential(*self.CreateClassifier())

    def CreateCNN(self):
        cnnlayers = []
        cnnlayers += [
            torch.nn.Conv2d(
                in_channels=self.nSensors,
                out_channels=16,
                kernel_size=(3, 5),
                padding=(1, 2),
            )
        ]
        cnnlayers += [torch.nn.ReLU()]
        cnnlayers += [
            torch.nn.MaxPool2d(kernel_size=(3, 5), stride=(3, 5), padding=(0, 0))
        ]
        cnnlayers += [
            torch.nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=(3, 5), padding=(1, 2)
            )
        ]
        cnnlayers += [torch.nn.ReLU()]
        cnnlayers += [
            torch.nn.MaxPool2d(kernel_size=(3, 5), stride=(3, 5), padding=(0, 0))
        ]
        cnnlayers += [
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 7), padding=(0, 3)
            )
        ]
        cnnlayers += [torch.nn.ReLU()]
        cnnlayers += [
            torch.nn.MaxPool2d(kernel_size=(1, 7), stride=(1, 7), padding=(0, 0))
        ]
        cnnlayers += [torch.nn.Flatten(start_dim=1, end_dim=-1)]
        return cnnlayers

    def CreateTCN(self):
        tcnlayers = []
        tcnlayers += [CausalConv1D(in_channels=384, out_channels=32, kernel_size=1)]
        tcnlayers += [
            cust_TCNLayer(in_channels=32, out_channels=32, kernel_size=2, dilation=1)
        ]
        tcnlayers += [
            cust_TCNLayer(in_channels=32, out_channels=32, kernel_size=2, dilation=2)
        ]
        tcnlayers += [
            cust_TCNLayer(in_channels=32, out_channels=32, kernel_size=2, dilation=4)
        ]
        return tcnlayers

    def CreateClassifier(self):
        classifier = []
        classifier += [torch.nn.Flatten(start_dim=1, end_dim=-1)]
        classifier += [torch.nn.Linear(32, 64)]
        classifier += [torch.nn.ReLU()]
        classifier += [torch.nn.Linear(64, 32)]
        classifier += [torch.nn.ReLU()]
        classifier += [torch.nn.Linear(32, self.nGestures)]
        return classifier

    def forward(self, x):
        cnnoutputs = []
        for i in range(self.nTimeSteps):
            cnnoutputs += [self.CNN(x[i])]
        tcninput = torch.stack(cnnoutputs, dim=2)
        tcnoutput = self.TCN(tcninput)
        classifierinput = tcnoutput.permute(0, 2, 1)
        outputs = []
        for i in range(self.nTimeSteps):
            outputs += [self.Classifier(classifierinput[:, i])]
        outputs = torch.stack(outputs, dim=1)
        return outputs.permute(1, 0, 2)


def compute_loss(outputs, labels, criterion):
    loss = 0
    for i in range(numberOfTimeSteps):
        loss += criterion(outputs[i], labels[i])
    return loss


def compute_acc(outputs, labels):
    # print(outputs.shape)
    pred = outputs.reshape(-1, numberOfGestures).max(1)
    squashed_labels = labels.reshape(-1)
    total = squashed_labels.shape[0]
    correct = pred[1].eq(squashed_labels).sum().item()
    return total, correct


def train(model, training_generator, epoch):
    # Training
    train_loss, correct, total = 0, 0, 0
    for batch, labels in training_generator:
        # Transfer to GPU
        batch, labels = batch.permute(1, 0, 2, 3, 4).to(device), labels.permute(
            1, 0
        ).to(device)

        optimizer.zero_grad()
        outputs = model(batch)
        loss = compute_loss(outputs, labels, criterion)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        curr_total, curr_correct = compute_acc(outputs, labels)
        total += curr_total
        correct += curr_correct

    acc = 100.0 * (correct) / total
    error_rate = 100.0 * (total - correct) / total
    lr = optimizer.param_groups[0]["lr"]
    tboard.add_scalar("loss/train", train_loss, epoch)
    tboard.add_scalar("error/train", error_rate, epoch)
    tboard.add_scalar("lr", lr, epoch)
    print("Epoch: " + str(epoch))
    print("Train -- Loss: %.3f | Acc: %.3f%% | LR: %e" % (train_loss, acc, lr))
    # print(epoch_loss)


def validate(model, val_generator, epoch):
    global max_val_acc

    val_loss, val_corr, val_tot = 0, 0, 0
    for batch, labels in val_generator:
        # Transfer to GPU
        batch, labels = batch.permute(1, 0, 2, 3, 4).to(device), labels.permute(
            1, 0
        ).to(device)

        outputs = model(batch)

        loss = compute_loss(outputs, labels, criterion)

        val_loss += loss.item()
        curr_total, curr_correct = compute_acc(outputs, labels)
        val_tot += curr_total
        val_corr += curr_correct

    val_acc = 100.0 * (val_corr) / val_tot
    print("Val -- Loss: %.3f | Acc: %.3f%%" % (val_loss, val_acc))
    # print(epoch_loss)

    if val_acc > max_val_acc:
        torch.save(model.state_dict(), f"./model_{validationPerson}_max.pt")
        max_val_acc = val_acc
        print("new max")

        loss, correct, total = 0, 0, 0


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
        print("No entries found for person with index 'p" + str(pdx) + "'")
        return

    return np.concatenate(SubjectData, axis=0), np.asarray(SubjectLabel)


def loadFeatures(
    pathtoDataset, listOfPeople, listofGestures, numberofInstanceCopies, lengthOfWindow
):
    personList = []
    pathToFeatures = pathToDataset + "data_feat/"
    for i in listOfPeople:
        personList.append(
            ParamList(
                i,
                pathToFeatures,
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
    # Load all previously extracted features
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

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    batch_size = 128
    max_epochs = 100

    model = Model(
        numberOfSensors,
        numberOfRangePointsPerSensor,
        lengthOfSubWindow,
        numberOfTimeSteps,
        numberOfGestures,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    tboard = SummaryWriter(log_dir="./logs", max_queue=2)
    tboard.add_graph(model, torch.rand(5, 32, 2, 32, 492).to(device))

    training_generator = torch.utils.data.DataLoader(
        traindataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    val_generator = torch.utils.data.DataLoader(
        valdataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    torch.cuda.empty_cache()

    for i in tqdm(range(max_epochs)):
        model.train()
        train(model, training_generator, i)
        model.eval()
        validate(model, val_generator, i)
