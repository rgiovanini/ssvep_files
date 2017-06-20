# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Renato Giovanini
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.io import loadmat

# Importar os arquivos .mat em arrays separados por frequencia
info_array_5hz = []
eeg_array_5hz = []
for i in range(1, 6):
    info_array_5hz.append(loadmat('5Hz/S{}_5Hz.mat'.format(i)))
    eeg_array_5hz.append(info_array_5hz[i-1]['X'])
eeg_array_5hz = np.array(eeg_array_5hz)

# utilizar como canal de interesse o ponto Oz (nº 14) e normalizar
Oz_array_5hz = np.array(eeg_array_5hz[:, :, 14])
Oz_array_5hz = Oz_array_5hz / np.amax(Oz_array_5hz)

# função para gerar o periodograma (periodograma dos raw signals)
fs = 256
periodogram_array = []
for i in range(5):
    periodogram_array.append(sig.periodogram(Oz_array_5hz[i, :], fs))
periodogram_array = np.array(periodogram_array)


# plotando todos os periodogramas
plt.figure(1)
for i in range(5):
    plt.plot(periodogram_array[i, 0, :], periodogram_array[i, 1, :],
             label='S{}'.format(i+1))
plt.legend()
plt.grid(True, linewidth=2)

# função para gerar o espectro de Welch
'''
welch_array = []
for i in range(5):
    welch_array.append(sig.welch(Oz_array_5hz[i, :], fs, nperseg=1000,
                                 noverlap=0.7, nfft=1024))
welch_array = np.array(welch_array)

plt.figure(2)
for i in range(5):
    plt.plot(welch_array[i, 0, :], welch_array[i, 1, :],
             label='S{}'.format(i+1))
'''

# projeto do banco de filtros na frequencia de interesse e na primeira
# harmonica (5, 10 Hz)
f_nyq = fs / 2
cutoff = np.array([4.8, 5.2, 9.8, 10.2, 14.8, 15.2, 19.8, 20.2]) / f_nyq
coeff = sig.firwin(401, cutoff, window='hamming', pass_zero=False)

# Resposta em frequência do filtro
w, h = sig.freqz(coeff, 1.)
plt.figure(2)
plt.plot((f_nyq/pi)*w, 20*np.log10(np.abs(h)))
plt.grid(True, linewidth=2)

# Aplicando o filtro ao array
Oz_array_5hz_filt = []
for i in range(5):
    Oz_array_5hz_filt.append(sig.lfilter(coeff, 1., Oz_array_5hz[i, :]))
Oz_array_5hz_filt = np.array(Oz_array_5hz_filt)

# periodograma dos sinais filtrados
Oz_array_filt_period = []
for i in range(5):
    Oz_array_filt_period.append(sig.periodogram(Oz_array_5hz_filt[i, :], fs))
Oz_array_filt_period = np.array(Oz_array_filt_period)

# plotando os periodogramas dos sinais filtrados
plt.figure(3)
for i in range(5):
    plt.plot(Oz_array_filt_period[i, 0, :], Oz_array_filt_period[i, 1, :])
plt.grid(True, linewidth=2)

# Método 1: trigger. Caso a PSD do sinal em uma das 4 bandas seja maior do que
# determinado valor (encontrado visualmente), o resultado é positivo

# Encontrar onde a PSD é maior do que um determinado valor (0.005)
ssvep_on = np.where(Oz_array_filt_period[:, 1, :] >= 0.005)

for i in range(5):
    if i in ssvep_on[0]:
        print("Saída ativada. O voluntário encontra-se sob estimulação de 5 Hz.")
    else:
        print("Saída não ativada. O voluntário não se encontra sob estimulação.")


