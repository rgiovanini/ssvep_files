#!/usr/bin/env python3

def design_butter_fir(cutoff, fs, order=101, band='pass', window='hamming'):
    '''
    FUNCTION DEFINITION
    '''
    import numpy as np
    import scipy.signal as sig

    fs = float(fs)
    fnyq = fs / 2
    cutoff = np.array(cutoff) / fnyq

    if order % 2 == 0:
        order = order + 1

    if band == 'pass':
        pass_zero = False
    elif band == 'stop':
        pass_zero = True
    else:
        print('ERROR')

    coeff = sig.firwin(order, cutoff, pass_zero=pass_zero, window=window)
    freq_vec, imp_vec = sig.freqz(coeff, 1.)

    return coeff, freq_vec, imp_vec


def generate_harmonics(freq, n_harmonics):
    '''
    FUNCTION DEFINITION
    '''
    import numpy as np
    
    harmonics = [(i * (j + 1)) for i in freq for j in range(n_harmonics)]
    harmonics = np.array(harmonics)

    return harmonics.reshape(len(freq), n_harmonics)


def generate_band_tolerance(freq, tolerance=0.3): #NEEDS FIXING TO CORRECT OUTPUT
    '''
    FUNCTION DEFINITION
    '''
    import numpy as np
    
    freq = np.array(freq)
    tol_array = np.hstack(np.array([[freq[i] - tolerance, freq[i] + tolerance]
                                   for i in range(len(freq))]))

    return tol_array

def apply_fir_filter(coefficients, input_array, axis=-1):
    '''
    FUNCTION DEFINITION
    '''
    import scipy.signal as sig
    
    return sig.lfilter(coefficients, 1., input_array, axis=axis)


