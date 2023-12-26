# ----------------------------------------------------------------------
#
# File: featureextraction.py
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


import numpy as np
from scipy.fftpack import fft, fftfreq, fftshift

def find_closest(val1, val2, target):
    return 0 if target - val1 >= val2 - target else -1
    
def get_closest_value_index(np_arr, target):
    """returns the index of the closest value to the target value in an ascending array"""
    n = np_arr.shape[0]
    left = 0
    right = n - 1
    mid = 0
    if target >= np_arr[n - 1]:
        return n - 1
    if target <= np_arr[0]:
        return 0

    while left < right:
        mid = (left + right) // 2  # find the mid
        if target < np_arr[mid]:
            right = mid
        elif target > np_arr[mid]:
            left = mid + 1
        else:
            return mid

    if target < np_arr[mid]:
        return mid + find_closest(np_arr[mid - 1], np_arr[mid], target)
    else:
        return mid + 1 + find_closest(np_arr[mid], np_arr[mid + 1], target)

def doppler_maps(x, take_abs=True, do_shift=True):
    x_len = x.shape[0]
    num_windows_per_instance = x.shape[1]
    time_wind = x.shape[2]
    num_range_points = x.shape[3]
    num_sensors = x.shape[4]

    if (take_abs):
        doppler = np.zeros((x_len, num_windows_per_instance, time_wind, num_range_points, num_sensors),
                           dtype=np.float32)  # take the absolute value, thus not complex data type
        for i_x in range(0, x_len):
            for i_instance in range(0, num_windows_per_instance):
                for i_range in range(0, num_range_points):
                    for i_sensor in range(0, num_sensors):
                        if (do_shift):
                            doppler[i_x, i_instance, :, i_range, i_sensor] = abs(
                                fftshift(fft(x[i_x, i_instance, :, i_range, i_sensor])))
                        else:
                            doppler[i_x, i_instance, :, i_range, i_sensor] = abs(
                                fft(x[i_x, i_instance, :, i_range, i_sensor]))

    else:
        doppler = np.zeros((x_len, num_windows_per_instance, time_wind, num_range_points, num_sensors),
                           dtype=np.complex64)  # complex value
        for i_x in range(0, x_len):
            for i_instance in range(0, num_windows_per_instance):
                for i_range in range(0, num_range_points):
                    for i_sensor in range(0, num_sensors):
                        if (do_shift):
                            doppler[i_x, i_instance, :, i_range, i_sensor] = fftshift(
                                fft(x[i_x, i_instance, :, i_range, i_sensor]))
                        else:
                            doppler[i_x, i_instance, :, i_range, i_sensor] = fft(
                                x[i_x, i_instance, :, i_range, i_sensor])
    return doppler


def center_of_mass_envelope(x):
    """returns the index of range value of the center of mass of the envelope.
    Also it calculates the center of mass of the left and right side of the center."""
    if (len(list(x.shape)) != 4):
        ValueError(
            "Input matrix has to have shape (x_len, tot_len_of_data_sample, num_range_points, num_sensors)! But got input matrix of shape: " + str(
                x.shape))
    x_len = x.shape[0]
    tot_len_of_data_sample = x.shape[1]
    num_range_points = x.shape[2]
    num_sensors = x.shape[3]
    com = np.zeros((x_len, tot_len_of_data_sample, num_sensors, 3), dtype=np.float32)

    for i_x in range(0, x_len):
        for i_sweep in range(0, tot_len_of_data_sample):
            for i_sens in range(0, num_sensors):
                temp = np.zeros(num_range_points)
                temp[0] = np.abs(x[i_x, i_sweep, 0, i_sens])
                for i_range in range(1, num_range_points):
                    temp[i_range] = temp[i_range - 1] + np.absolute(x[i_x, i_sweep, i_range, i_sens])
                center = get_closest_value_index(temp, temp[-1] / 2)
                com[i_x, i_sweep, i_sens, 0] = center
                com[i_x, i_sweep, i_sens, 1] = get_closest_value_index(temp[:center + 1],
                                                                       temp[center] / 2)  
                com[i_x, i_sweep, i_sens, 2] = center + get_closest_value_index(temp[center:], (
                        temp[center] + temp[-1]) / 2)  # get right side center
    return com

def av_energy(x):
    """returns the average energy over the time window or over the range per time window.
    It takes the absolute value of the returns for calculation of the energy"""
    if (len(list(x.shape)) != 5):
        raise ValueError("shape of input array to funct av_energy not correct, input shape was" + str(x.shape))
    x_len = x.shape[0]
    num_windows_per_instance = x.shape[1]
    time_wind = x.shape[2]
    num_range_points = x.shape[3]
    num_sensors = x.shape[4]

    # average energy in time
    av_en_time = np.zeros((x_len, num_windows_per_instance, time_wind, num_sensors),
                          dtype=np.float32)  # sum up over the range points
    # average energy in range
    av_en_range = np.zeros((x_len, num_windows_per_instance, num_range_points, num_sensors),
                           dtype=np.float32)  # sum up over the time points

    for i_x in range(0, x_len):
        # if (i_x % 200 == 0):
        #     print("percentage done: ", 100 * i_x / float(x.shape[0]))
        for i_windows in range(num_windows_per_instance):
            for i_sens in range(0, num_sensors):
                av_en_time[i_x, i_windows, :, i_sens] = np.abs(x[i_x, i_windows, :, :, i_sens]).sum(axis=-1)
                av_en_range[i_x, i_windows, :, i_sens] = np.abs(x[i_x, i_windows, :, :, i_sens]).sum(axis=0)
    return av_en_time, av_en_range

def compl_signal_variation(x, freq):
    """get average variance of complex signal.
    From the complex signal variance the variance in amplitude and phase can be extracted by np.abs(diff_arr) and np.angle(diff_arr)"""
    # get average variance of the complex signal
    x_len = x.shape[0]
    num_windows_per_instance = x.shape[1]
    time_wind = x.shape[2]
    num_range_points = x.shape[3]
    num_sensors = x.shape[4]

    diff_arr = np.zeros((x_len, num_windows_per_instance, time_wind - 1, num_range_points, num_sensors),
                        dtype=np.complex64)

    for i_x in range(0, x_len):
        # if (i_x % 200 == 0):
        #     print("percentage done: ", 100 * i_x / float(x.shape[0]))
        for i_windows in range(num_windows_per_instance):
            for i_range in range(0, num_range_points):
                for i_sensor in range(0, num_sensors):
                    for i_diff in range(0, time_wind - 1):
                        diff_arr[i_x, i_windows, i_diff, i_range, i_sensor] = (x[
                                                                                   i_x, i_windows, i_diff + 1, i_range, i_sensor] -
                                                                               x[
                                                                                   i_x, i_windows, i_diff, i_range, i_sensor]) / (
                                                                                      1 / freq)  # 1/freq is the timestep

    diff_arr_sum_over_time = diff_arr.sum(axis=-2)
    diff_arr_sum_over_range = diff_arr.sum(axis=-3)
    return diff_arr, diff_arr_sum_over_time, diff_arr_sum_over_range

