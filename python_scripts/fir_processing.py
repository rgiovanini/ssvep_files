# imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.io import loadmat

# functions definitions


def design_butter_fir(cutoff, fs, order=101, band='pass', window='hamming'):
    '''
    FUNCTION DESCRIPTION
    '''

    fs = float(fs)
    fn = fs / 2.
    cutoff = np.array(cutoff) / fn

    # checking the band type (pass/reject)
    if band == 'pass':
        pass_zero = False
    elif band == 'stop':
        pass_zero = True
    else:
        print('ERROR: Argument', band, 'unknown.')

    # filter design
    coeff = sig.firwin(order, cutoff, pass_zero=pass_zero, window=window)
    freq_vec, impulse_vec = sig.freqz(coeff, 1.)

    return coeff, freq_vec, impulse_vec


def generate_harmonics(freq, n_harmonics):
    '''
    FUNCTION DESCRIPTION
    '''

    harmonics = [(i * (j + 1)) for i in freq for j in range(n_harmonics)]
    harmonics = np.array(harmonics)

    return harmonics.reshape(len(freq), n_harmonics)


def generate_band_tolerance(freq, tolerance=0.3):  # ONLY FOR 1D ARRAYS FOR NOW
    '''
    FUNCTION DESCRIPTION
    '''

    freq = np.array(freq)
    tol_array = np.hstack(np.array([[freq[i] - tolerance, freq[i] + tolerance]
                                   for i in range(len(freq))]))

    return tol_array


if __name__ == '__main__':

    # loading and taking eeg signals into data and target vectors
    path = '/home/renato/Dropbox/Mestrado/final/kolodziej_dataset/dataset_15_canais.mat'
    eeg = loadmat(path)
    data = eeg['data']
    target = eeg['target']

    # sampling and nyquist frequencies
    fs = 256.
    fn = fs / 2.

    # design the four filter banks
    harmonics = generate_harmonics([5, 6, 7, 8], 4)
    bands = [generate_band_tolerance(harmonics[i], 0.3)
                                    for i in range(len(harmonics))]

    filter_pack = [design_butter_fir(cutoff=i, fs=fs, order=401) for i in bands]
