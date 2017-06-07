#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:47:12 2017

@author: renato

Denoising de um sinal utilizando a Transformada Wavelet Discreta
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import periodogram
from sklearn.preprocessing import scale
import pywt

#carregando os dados de eeg a serem filtrados 
fs = 256.
data = loadmat('/media/renato/Dados/Mestrado/SSVEP-Datasets/Kolodziej/all_data/S1_5Hz.mat')
eeg = data['X']
eeg = np.transpose(eeg)
y = scale(eeg[0])
t = np.arange(len(eeg[0])) / fs

coeffs = pywt.wavedec(y, pywt.Wavelet('db4'), level=2)
plt.figure(1)
plt.plot(periodogram(coeffs[0],fs=32)[0], periodogram(coeffs[0],fs=128)[1])
#plt.figure(2)
#plt.plot(periodogram(coeffs[1],fs=128)[0], periodogram(coeffs[1],fs=128)[1])

#setando os detalhes em zero. 
coeffs = [coeffs[0], np.zeros(len(coeffs[1])), np.zeros(len(coeffs[2]))]
y_rec = pywt.waverec(coeffs, pywt.Wavelet('db4'))
