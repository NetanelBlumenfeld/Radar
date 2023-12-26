# ----------------------------------------------------------------------
#
# File: extract_5G.py
#
# Last edited: 02.02.2021        
# 
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
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


from scipy.fftpack import fft, fftfreq, fftshift
import numpy as np
import os

from featureextraction import *
from listutils import *

import re

gestures = ["FingerSlider", "SwipeRL", "PushDown", "PullUp", "PalmTilt"]

sessions = list(range(0,9))

numpyPath = './'
featurePath = './'

instances = 50

freq = 256

datasetPath = "../5G/"
DataSubdir = "data/"
numpyDataSubdir = "data_npy/" 

minSweeps = 32
numWin = 3

rangePoints = 414

def extraction(gestures, sessions, instances):
    for gdx, gestureName in enumerate(gestures):
        print('\n')
        print('gesture name: ', gestureName)
        gestPath = gestureName + "/"
        dataGesture = np.array([])
        
        for sdx in sessions:
            pathComponentSession = "session_" + str(sdx) + "/"
            
            for idx in range(0, instances):    
                path = datasetPath + DataSubdir + gestPath + pathComponentSession + str(idx) + "_" + gestureName + ".txt"
                #print(path)
                #print(os.path.exists(path))
                lines = lines2list(path)
                
                numberOfSweeps = 0
                for i in reversed(lines):
                    if 'usecs_tot' in i:
                        num = re.search("usecs_tot: [0-9]*,", i)
                        numspan = [num.span()[0]+11, num.span()[1]-1]
                        intnum = int(i[numspan[0]:numspan[1]])
                        usecs_tot = intnum
                        
                    if 'Sweep' in i:
                        num = re.search(" [0-9]*,", i)
                        numspan = [num.span()[0]+1, num.span()[1]-1]
                        intnum = int(i[numspan[0]:numspan[1]])
                        numberOfSweeps = intnum+1
                        break
                    
                sweepFrequency = freq # numberOfSweeps / usecs_tot * 1e6

                if (minSweeps > numberOfSweeps):
                    continue

                dataStacked = []
                for i in lines:
                    if "D" in i:
                        A = re.search("A: [0-9]*\.[0-9]*,", i)
                        Aspan = [A.span()[0]+3, A.span()[1]-1]
                        intA = float(i[Aspan[0]:Aspan[1]])
                        Amplitude = intA

                        P = re.search("P: .*", i)
                        Pspan = [P.span()[0]+3, P.span()[1]-1]
                        intP = float(i[Pspan[0]:Pspan[1]])
                        Phase = intP

                        dataStacked.append(Amplitude * np.exp(1j*Phase))

                dataStacked = np.array(dataStacked)
                dataStacked = dataStacked.reshape(numberOfSweeps, rangePoints, 1)
                numSweeps = int(sweepFrequency*1+0.5)

                data = np.zeros((numWin, numSweeps, rangePoints, 1), dtype=np.complex64)
                 
                difference = numberOfSweeps - numSweeps
                if (difference < 0):
                    for wdx in range(0, numWin):
                        data[wdx, :numberOfSweeps, :, :] = dataBinaryStacked
                else:
                    windowStartIndices = [int(i*difference/numWin) for i in range(0,numWin)]
                    for wdx in range(0, numWin):
                        data[wdx, :, :, :] = dataStacked[windowStartIndices[wdx]:(windowStartIndices[wdx]+numSweeps)]
                if (0 < dataGesture.size):
                    dataGesture = np.vstack((dataGesture, data))
                else:
                    dataGesture = data
                    
        persPath = "p0/"
        pathOutputNumpy = datasetPath + numpyDataSubdir + persPath + gestureName + "_1s.npy"
        ensure_dir(pathOutputNumpy)
        np.save(pathOutputNumpy, dataGesture)

extraction(gestures, sessions, instances)
