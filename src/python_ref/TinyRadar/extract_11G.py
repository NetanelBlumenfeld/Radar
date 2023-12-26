# ----------------------------------------------------------------------
#
# File: extract_11G.py
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


# from scipy.fftpack import fft, fftfreq, fftshift
import numpy as np
import os

from featureextraction import *
from listutils import *

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
    "RandomGesture",
]

# gestures = ["FingerSlide", "SlowSwipe", "Push", "Pull", "PalmTilt"]

numpyPath = "./"
featurePath = "./"

singleuserlist = list(map(lambda x: "0_" + str(x), list(range(1, 21, 1))))

people = list(range(1, 26, 1))

people += singleuserlist

sessions = list(range(0, 5))

instances = 7

# instances = 50

freq = 160

freq = 256

datasetPath = "/Users/netanelblumenfeld/Downloads/11G/"

# datasetPath = "./5G/"

binaryDataSubdir = "data/"
numpyDataSubdir = "data_npy/"

minSweeps = 32
numWin = 3


def extraction(people, gestures, sessions, instances):
    for gdx, gestureName in enumerate(gestures):
        print("\n")
        print("gesture name: ", gestureName)
        gestPath = gestureName + "/"
        for pdx, person in enumerate(people):
            print("p:", person, " ", end="", flush=True)

            dataGesture = np.array([])

            persPath = "p" + str(person) + "/"
            for sdx in sessions:
                pathComponentSession = "sess_" + str(sdx) + "/"
                for idx in range(0, instances):
                    basePath = (
                        datasetPath
                        + binaryDataSubdir
                        + persPath
                        + gestPath
                        + pathComponentSession
                        + str(idx)
                    )
                    sensorPath0 = basePath + "_s0.dat"
                    sensorPath1 = basePath + "_s1.dat"
                    infoPath = basePath + "_info.txt"
                    info = lines2list(infoPath)

                    numberOfSweeps = int(info[1])
                    sweepFrequency = int(info[2])
                    sensorRangePoints0 = int(info[14])
                    sensorRangePoints1 = int(info[15])
                    if minSweeps > numberOfSweeps:
                        continue

                    dataBinarySensor0 = np.fromfile(sensorPath0, dtype=np.complex64)
                    dataBinarySensor1 = np.fromfile(sensorPath1, dtype=np.complex64)

                    dataBinarySensor0 = dataBinarySensor0.reshape(
                        (numberOfSweeps, sensorRangePoints0)
                    )
                    dataBinarySensor1 = dataBinarySensor1.reshape(
                        (numberOfSweeps, sensorRangePoints1)
                    )

                    dataBinaryStacked = np.stack(
                        (dataBinarySensor0, dataBinarySensor1), axis=-1
                    )

                    numSweeps = sweepFrequency * 1

                    data = np.zeros(
                        (numWin, numSweeps, sensorRangePoints0, 2), dtype=np.complex64
                    )

                    difference = numberOfSweeps - numSweeps
                    if difference < 0:
                        for wdx in range(0, numWin):
                            data[wdx, :numberOfSweeps, :, :] = dataBinaryStacked
                    else:
                        windowStartIndices = [
                            int(i * difference / numWin) for i in range(0, numWin)
                        ]
                        for wdx in range(0, numWin):
                            data[wdx, :, :, :] = dataBinaryStacked[
                                windowStartIndices[wdx] : (
                                    windowStartIndices[wdx] + numSweeps
                                )
                            ]
                    if 0 < dataGesture.size:
                        dataGesture = np.vstack((dataGesture, data))
                    else:
                        dataGesture = data

            pathOutputNumpy = (
                datasetPath + numpyDataSubdir + persPath + gestureName + "_1s.npy"
            )
            ensure_dir(pathOutputNumpy)
            print(pathOutputNumpy)
            np.save(pathOutputNumpy, dataGesture)
            break


extraction(people, gestures, sessions, instances)
