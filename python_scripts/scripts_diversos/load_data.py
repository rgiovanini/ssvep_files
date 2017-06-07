#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:54:31 2017

@author: renato
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pywt as wt
from cesium import featurize
import scipy.stats as stats


# carregando todos os meus dados de 5 Hz
data_5hz = np.array([loadmat('5Hz/S{}_5Hz.mat'.format(m+1)) for m in range(5)])

# reshape das time series dos sinais de eeg em 'X' e criando os targets em '5'

for m in range(5):
    data_5hz[m]['X'] = np.transpose(data_5hz[m]['X'])
    data_5hz[m]['target'] = np.array([5]*len(data_5hz[0]['X']))

# definindo as features utilizadas em Subasi (2005)
def mean_signal(t, m, e):
    return np.mean(m)

def std_signal(t, m, e):
    return np.std(m)

def mean_square_signal(t, m, e):
    return np.mean(m ** 2)

def abs_diffs_signal(t, m, e):
    return np.sum(np.abs(np.diff(m)))

def skew_signal(t, m, e):
    return stats.skew(m)

# criando um dict com as features que acabei de fazer
features = {
    'mean':mean_signal,
    'std':std_signal,
    'mean_square':mean_square_signal,
    'abs_diffs':abs_diffs_signal,
    'skew':skew_signal
}

# Usando uma DWT de 4 niveis, obtenho as sub-bandas cA4, cD4, cD3, cD2, cD1 dos espectros dos sinais de cada canal 
features_5hz = []
for m in range(5):
    data_5hz[m]['DWTs'] = [wt.wavedec(i, wt.Wavelet('db2'), level=4) for i in data_5hz[m]['X']]
    # Obtendo as features baseado nos coeficientes da DWT obtidos
    features_5hz = featurize.featurize_time_series(times=None, values=data_5hz[m]['DWTs'], errors=None, 
                                                       features_to_use=list(features.keys()), custom_functions=features)    
    
