#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:43:51 2017

@author: renato
"""

import numpy as np
from scipy.io import loadmat, savemat

# carregando os canais de interesse dos arquivos .mat e gerando o array
lista_canais = list(range(16))#[0, 3, 4, 9, 10, 11, 12, 13, 14, 15]
n_canais = len(lista_canais)
path = '/home/renato/Dropbox/Mestrado/final/kolodziej_dataset/all_data/S{}_{}Hz.mat'
data = [np.transpose(loadmat(path.format(i, j))['X'])[lista_canais]
        for j in range(5, 9) for i in range(1, 6)]
data = np.vstack(data)

# criando o array de targets
target = [np.full(n_canais * 5, i, dtype='int64') for i in range(5, 9)]
target = np.hstack((target[0], target[1], target[2], target[3]))

# salvando o dataset
# dataset_dict = {'data': data,
#                'target': target,
#                'fs': 256.}
# save_path = '/home/renato/Dropbox/Mestrado/programas_final/kolodziej_dataset/dataset_7_canais.mat'
# savemat(save_path, dataset_dict)

# whitening dos dados de entrada (standardization)
from sklearn.preprocessing import scale

data_scaled = np.array([scale(data[i]) for i in range(len(data))])

# criando as features baseadas em Guo et al e na decomposição por DWT

import pywt

# decompondo em sub-bandas usando a DWT e wavelet db12 e tomando
# apenas as aproximações
n_levels = 4
data_scaled_dwt = [pywt.wavedec(data_scaled[i], pywt.Wavelet('db12'),
                   level=n_levels)[0] for i in range(len(data_scaled))]
data_scaled_dwt = np.vstack(data_scaled_dwt)

# treinamento do classificador
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

acc_list = []

# repetições de separação, treinamento e predição
for i in range(10):
    train, test = train_test_split(np.arange(len(target)), test_size=0.8)
    clf_RF = RandomForestClassifier(n_estimators=1500, max_features='auto',
                                    n_jobs=-1, bootstrap=True,
                                    warm_start=0, oob_score=True)
    clf_RF.fit(data_scaled_dwt[train], target[train])
    # predição
    from sklearn.metrics import accuracy_score

    preds_clw_RF = clf_RF.predict(data_scaled_dwt)

    # teste de acurácia
    print("Wavelet transform features using RF: training accuracy={:.2%},"
          "test accuracy={:.2%}".format(
              accuracy_score(preds_clw_RF[train], target[train]),
              accuracy_score(preds_clw_RF[test], target[test])))
    acc_list.append(accuracy_score(preds_clw_RF[test], target[test]))

print("mean test accuracy: {:.2%} ".format(np.mean(acc_list)))
print("mean standard deviation: {:.2%} ".format(np.std(acc_list)))
