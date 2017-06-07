# -*- coding: utf-8 -*-

"""
vim Editor
@author: Renato Giovanini

This script performs the EEG data processing based on Kolodziej's EEG dataset
and filter bank method.
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.io import loadmat
#import timing

def select_eeg_all():
    '''
    Import all eeg files to an array
    '''
    array_all_data = []
    array_all_eeg = []
    for i in range(1, 6):
        for j in range(5, 9):
            array_all_data.append(loadmat(
                'all_files/S{}_{}Hz.mat'.format(i, j)))
            array_all_eeg.append(array_all_data[i-1]['X'])
    return np.array(array_all_eeg)


def select_eeg_channel(input_array, channel):
    '''Select only the channel of interest
    '''
    array_shape = input_array.shape
    if len(array_shape) == 4:
        return np.array(input_array[:, :, :, channel])
    elif len(array_shape) == 3:
        return np.array(input_array[:, :, channel])


def select_eeg_frequency(freq):
    '''Select only the channels with the tests with the specified frequency
    '''
    array_freq = []
    array_eeg_freq = []
    file_string = 'all_files/S{}_{}Hz.mat'
    for i in range(1, 6):
        for j in range(5, 9):
            formatted_string = file_string.format(i, j)
            temp = loadmat(str(formatted_string))
            if formatted_string[13] == str(freq):
                array_freq.append(temp)
    for i in range(4):
        array_eeg_freq.append(array_freq[i]['X'])
    return np.array(array_eeg_freq)


def select_eeg_subject(subject):
    '''Select only the tests with the desired subject.
    '''
    array_subject = []
    array_eeg_subject = []
    file_string = 'all_files/S{}_{}Hz.mat'
    for i in range(1, 6):
        for j in range(5, 9):
            formatted_string = file_string.format(i, j)
            temp = loadmat(str(formatted_string))
            if formatted_string[11] == str(subject):
                array_subject.append(temp)
    for i in range(4):
        array_eeg_subject.append(array_subject[i]['X'])
    return np.array(array_eeg_subject)


def design_fir_butter(cutoff, f_sampl, order=101, band='pass'):
    '''Design a butterworth filter with specified order. If the order is not
    specified, the value is 101. The band parameter options is 'pass' and 'stop'
    '''
    f_sampl = float(f_sampl)
    f_nyq = f_sampl / 2                   # Nyquist frequency
    cutoff = np.array(cutoff) / f_nyq     # Normalizing the cutoff frequency vector
    if order % 2 == 0:                    # ensures that the order is odd
        order = order + 1

    if band == 'pass':                    # defines DC gain (see signal.firwin
        pass_zero = False                 # documentation for details)
    elif band == 'stop':
        pass_zero = True
    else:
        print('ERROR: argument', band, 'not known')
    coeff = sig.firwin(order, cutoff, pass_zero=pass_zero, window='hamming')

    freq_vec, imp_vec = sig.freqz(coeff, 1.)  # Frequency response of the filter

    return coeff, freq_vec, imp_vec


def apply_fir_filter(coeff, input_array, axis):
    '''Apply filter coefficients to an input array. It is mathematically
    equivalent to make a convolution sum between the two systems.
    '''
    return sig.lfilter(coeff, 1., input_array, axis=axis)


def normalize_array(input_array):
    '''Return the normalized values of array. The norm is the max. value.
    '''
    return input_array / np.amax(input_array)


def generate_band_tolerance(freq, tol):
    '''generates an array with the right cutoff frequencies for a bandpass or
    band reject filter, based on the desired tolerancy.
    '''
    freq = np.array(freq)
    f1 = freq - tol
    f2 = freq + tol
    freq_tol = []
    for i in range(len(f1[0])):
        for j in range(len(f2[0])):
            freq_tol.append([f1[i][j], f2[i][j]])
    return np.array(freq_tol)
   # return np.transpose(np.array([freq-tol, freq+tol])).reshape(2*len(freq))


def generate_harmonics(input_freq, number_of_harmonics):
    '''Generate the desired harmonics based on the specified number
    '''
    harmonics = []
    for i in input_freq:
        for j in range(number_of_harmonics):
            harmonics.append(i*(j+1))
    harmonics = np.array(harmonics)
    return harmonics.reshape(len(input_freq), number_of_harmonics)


def test(input_array, trigger):
    '''Tests the input array and returns the positive and negative results for
    the tests, based on the trigger value
    '''
    positive, negative = [], []
    p, n = 0, 0
    for i in range(4):
        filter_bank = input_array[i]
        tests = np.array(np.where(filter_bank >= trigger))
        for j in range(5):
            if j in tests[0]:
                p = p + 1
        for k in range(len(set(tests[1]))):
            if k in set(tests[1]) != i:
                n = n + 1
        positive.append(p)
        negative.append(n)
        p, n = 0, 0
    return positive, negative


if __name__ == "__main__":

    # Load dataset by subject and normalize
    subjects_array = []
    for i in range(5):
        subjects_array.append(select_eeg_subject(i+1))
    subjects_array = normalize_array(np.array(subjects_array))
    Oz_array = subjects_array[:, :, :, 15]

    # Make the array of the estimated power spectrum densities
    fs = 256.       #Sampling frequency
    fnyq = fs / 2.  # Nyquist frequency
    Oz_period = sig.periodogram(Oz_array, fs, window='hamming', axis=2)
    Oz_period = np.array(Oz_period)
    #plt.plot(Oz_period[0], Oz_period[1][0,0,:]


    # Design the filterbank (5, 6, 7, 8 Hz and its four harmonics)
    freq_array = generate_harmonics([5, 6, 7, 8], 4)
    cutoff_array = generate_band_tolerance(freq_array, 0.3)
    cutoff_groups = []
    for i in range(4):
        cutoff_groups.append(cutoff_array[4*i:4*i+4])
    cutoff_groups = np.array(cutoff_groups)

    filter_parameters = []
    for i in range(4):
        filter_parameters.append(design_fir_butter(cutoff_groups[i].reshape(
            np.size(cutoff_groups[i])), fs, order=401))
    filter_parameters = np.array(filter_parameters)

    # plt.plot((fnyq/pi)*filter_parameters[0,1], 20*np.log10(np.abs(
    #     filter_parameters[0,2])))

    # Apply filters to signals
    filtered_signal = []
    coefficients = filter_parameters[:,0]
    '''
    # Subject 1:
    S1 = Oz_array[0]
    for i in range(4):   #para cada frequencia
       for j in range(4):   #para cada banco de filtros
          filtered_signal.append(apply_fir_filter(coefficients[j], S1[i]))
    filtered_signal = np.array(filtered_signal)

    # Estimate the PSD of the filtered signal

    # Subject b1:
    filtered_period = np.array(sig.periodogram(filtered_signal, fs,
                                               window='hamming', axis=1))
                                              '
    # Apply trigger method
    ssvep_on = np.where(filtered_period[1] >= 0.005)
    print("Subject 1: number of positive results: ", np.size(ssvep_on[0]))
    print("Number of expected positive results: ", 4)

    '''
    # Generalizando para todos os subjects
    Oz_array_filtered = []
    for i in range(4):  #para cada banco de filtros
        Oz_array_filtered.append(apply_fir_filter(coefficients[i], Oz_array,
                                                  axis=2))
    Oz_array_filtered = np.array(Oz_array_filtered)

    Oz_period_filtered = np.array(sig.periodogram(Oz_array_filtered, fs,
                                                  window='hamming', axis=3))

    test_array = Oz_period_filtered[1]
    positive, negative = test(test_array, 0.002)




