def design_butter_fir(cutoff, fs, order=101, band='pass', window='hamming'):
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
    import numpy as np
    harmonics = [(i * (j + 1)) for i in freq for j in range(n_harmonics)]
    harmonics = np.array(harmonics)

    return harmonics.reshape(len(freq), n_harmonics)


def generate_band_tolerance(freq, tol):
    import numpy as np
    freq = np.array(freq)
    f1 = freq + tol
    f2 = freq - tol
    freq_tol = [[f1[i][j], f2[i][j]] for i in range(len(f1[0])
                                     for j in range(len(f2[0])))]

    return np.array(freq_tol)
