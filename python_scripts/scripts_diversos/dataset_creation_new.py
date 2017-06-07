#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:43:51 2017

@author: renato
"""

import numpy as np
from scipy.io import loadmat, savemat
import pandas as pd

# carregando os canais de interesse dos arquivos .mat e gerando o array
lista_canais = [0,3,4,9,12,14,15]
path = '/home/renato/Dropbox/Mestrado/programas_final/kolodziej_dataset/all_data/S{}_{}Hz.mat'
all_data = [np.transpose(loadmat(path.format(i,j))['X'])[lista_canais] 
            for j in range(5,9) for i in range(1,6)]
all_data = np.vstack(all_data)

# criando o array de targets 
target = [np.full(7*5, i, dtype='int64') for i in range(5,9)]
target = np.hstack((target[0], target[1], target[2], target[3]))

# salvando o dataset
dataset_dict = {'data': all_data,
                'target': target,
                'fs': 256.}
save_path = '/home/renato/Dropbox/Mestrado/programas_final/kolodziej_dataset/dataset_7_canais.mat'
savemat(save_path, dataset_dict)

