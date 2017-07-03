#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: renato
"""
# imports
import numpy as np
from scipy.io import loadmat
import funcoes_filtragem as filt


# loading and taking eeg signals into data and target vectors
path = '/home/renato/Dropbox/Mestrado/final/kolodziej_dataset/dataset_15_canais.mat'
eeg = loadmat(path)
data = eeg['data']
target = eeg['target'].reshape(eeg['target'].shape[1])

# scaling instances
from sklearn.preprocessing import scale
data_scaled = np.array([scale(data[i]) for i in range(len(data))])

# sampling and nyquist frequencies
fs = 256.
fn = fs / 2.

# design the four filter banks
harmonics = filt.generate_harmonics([5, 6, 7, 8], 4)
bands = [filt.generate_band_tolerance(harmonics[i], 0.3) 
                                for i in range(len(harmonics))]

filter_pack = np.array([filt.design_butter_fir(cutoff=i, fs=fs, order=401) 
                                for i in bands])

############################################################################    
# AQUI CABE UMA DECIMAÇÃO ANTES DE APLICAR
############################################################################

# applying all the instances to all filters
data_scaled_filt = [filt.apply_fir_filter(filter_pack[i][0], data_scaled)
                                    for i in range(4)]

# saving data array 
save_path = '/home/renato/Dropbox/Mestrado/final/python_scripts/data_scaled_filt.npy'
np.save(save_path, data_scaled_filt)
    
    
    

