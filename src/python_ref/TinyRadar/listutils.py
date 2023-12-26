# ----------------------------------------------------------------------
#
# File: listutils.py
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

def lines2list(pathToFile):
    with open(pathToFile, "r") as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

def get_nested_list4d(num_persons, num_sessions, num_gestures):
    data_l = list() 
    for pers in range(0, num_persons):
        person = list()
        for sess in range(0, num_sessions):
            sessions = list()
            for gest in range(0, num_gestures):
                np_arrs = list()
                sessions.append(np_arrs)
            person.append(sessions)
        data_l.append(person)
    return data_l


def get_nested_list2d(num_gestures):
    ret_list = list() 
    for gest in range(0, num_gestures):
        gest = list()
        ret_list.append(gest)
    return ret_list

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)



