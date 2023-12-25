# ----------------------------------------------------------------------
#
# File: extract_numpy2feat.py
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


import os

import numpy as np
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

numpyFiles = "/Users/netanelblumenfeld/Downloads/11G/data_npy/"
featureFiles = "/Users/netanelblumenfeld/Downloads/11G/data_fit/"

people = list(range(1, 26, 1))
singleuserlist = list(map(lambda x: "0_" + str(x), list(range(1, 21, 1))))

freq = 160

windowSize = 32

people += singleuserlist

gestures = ["FingerSlider", "SwipeRL", "PushDown", "PullUp", "PalmTilt"]

people = [1]
freq = 256
windowSize = 32


def extraction(people, gestures, windowLength):
    for idx in people:
        print("Person:", idx)
        for nameGesture in gestures:
            print("Gesture: {} ".format(nameGesture), end="", flush=True)

            numpyFile = numpyFiles + "p" + str(idx) + "/" + nameGesture + "_1s.npy"
            featureFile = featureFiles + "p" + str(idx) + "/"
            featureName = nameGesture + "_1s_"
            dataGesture = np.load(numpyFile)

            numberOfWindows = dataGesture.shape[0]
            numberOfSweeps = dataGesture.shape[1]
            numberOfRangePoints = dataGesture.shape[2]
            numberOfSensors = dataGesture.shape[3]

            ensure_dir(featureFile)

            com = center_of_mass_envelope(dataGesture)
            np.save(featureFile + featureName + "com.npy", com)

            numberOfSubWindows = int(numberOfSweeps / windowLength)

            dataGesture = dataGesture.reshape(
                (
                    numberOfWindows,
                    numberOfSubWindows,
                    windowLength,
                    numberOfRangePoints,
                    numberOfSensors,
                )
            )

            filenameFeature = featureFile + featureName + "wl" + str(windowLength) + "_"

            doppl = doppler_maps(dataGesture, take_abs=True)
            np.save(filenameFeature + "doppl.npy", doppl)

            av_en_time, av_en_range = av_energy(dataGesture)
            np.save(filenameFeature + "av_en_t.npy", av_en_time)
            np.save(filenameFeature + "av_en_r.npy", av_en_range)

            (
                diff_arr,
                diff_arr_sum_over_time,
                diff_arr_sum_over_range,
            ) = compl_signal_variation(dataGesture, freq=freq)
            np.save(filenameFeature + "diff_arr.npy", diff_arr)
            np.save(filenameFeature + "diff_arr_sot.npy", diff_arr_sum_over_time)
            np.save(filenameFeature + "diff_arr_sor.npy", diff_arr_sum_over_range)


extraction(people, gestures, windowSize)
