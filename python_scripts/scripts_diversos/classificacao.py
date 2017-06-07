#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 09:04:49 2017

@author: renato
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd

data = loadmat('/media/renato/Dados/Mestrado/SSVEP-Datasets/Kolodziej/dataset_kolodziej.mat')
# modificando a key 'data' para deixar semelhante à do exemplo
data['data'] = np.vsplit(data['data'],data['data'].shape[0])
data['data'] = [np.reshape(data['data'][i],data['data'][i].shape[1]) for i in range(len(data['data']))]
data['label'] = data['label'].reshape(data['label'].shape[1])

# whitening 
from sklearn import preprocessing

x = data['data'][0]
x_scaled = preprocessing.scale(x)
data['data'] = [preprocessing.scale(data['data'][m]) for m in range(len(data['data']))]


from cesium import featurize
import pywt
import scipy.stats

def mean_signal(t, m, e):
    return np.mean(m)

def std_signal(t, m, e):
    return np.std(m)

def mean_square_signal(t, m, e):
    return np.mean(m ** 2)

def abs_diffs_signal(t, m, e):
    return np.sum(np.abs(np.diff(m)))

def skew_signal(t, m, e):
    return scipy.stats.skew(m)

guo_features = {
    'mean': mean_signal,
    'std': std_signal, 
    'mean2': mean_square_signal, 
    'abs_diffs': abs_diffs_signal, 
    'skew': skew_signal
}

fset_guo = featurize.featurize_time_series(times=None, 
                                           values=data['data'],
                                           errors=None,
                                           features_to_use=list(guo_features.keys()),
                                           custom_functions=guo_features)

#print(fset_guo.head())

# Dividindo o sinal em sub-bandas através da DWT
import pywt

n_subbands = 5
data['dwt'] = [pywt.wavedec(m, pywt.Wavelet('db6'), level=n_subbands-1) for m in data['data']]
fset_dwt = featurize.featurize_time_series(times=None, 
                                           values=data['dwt'],
                                           errors=None,
                                           features_to_use=list(guo_features.keys()),
                                           custom_functions=guo_features)
#print(fset_dwt.head())

# 5 Sub-bandas: A4, D4, D3, D2, D1

# Construção do Modelo de classificação 
# testes com classificadores KNN e RF


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

train, test = train_test_split(np.arange(len(data['label'])))
model_RF_guo = RandomForestClassifier(n_estimators=128, max_features='auto', random_state=0)
model_RF_guo.fit(fset_guo.iloc[train], data['label'][train])

model_KNN_guo = KNeighborsClassifier(3)
model_KNN_guo.fit(fset_guo.iloc[train], data['label'][train])

model_RF_dwt = RandomForestClassifier(n_estimators=1800, max_features='auto',n_jobs=-1,
                                      random_state=0,bootstrap=True, warm_start=0,
                                      oob_score=True)
model_RF_dwt.fit(fset_dwt.iloc[train], data['label'][train])


# Predição 
from sklearn.metrics import accuracy_score

preds_RF_guo = model_RF_guo.predict(fset_guo)
preds_KNN_guo = model_KNN_guo.predict(fset_guo)
preds_RF_dwt = model_RF_dwt.predict(fset_dwt)

print("Guo et al. features using RF: training accuracy={:.2%}, test accuracy={:.2%}".format(
          accuracy_score(preds_RF_guo[train], data['label'][train]),
          accuracy_score(preds_RF_guo[test], data['label'][test])))
print("Guo et al. features using KNN: training accuracy={:.2%}, test accuracy={:.2%}".format(
          accuracy_score(preds_KNN_guo[train], data['label'][train]),
          accuracy_score(preds_KNN_guo[test], data['label'][test])))
print("Wavelet transform features using RF: training accuracy={:.2%}, test accuracy={:.2%}".format(
          accuracy_score(preds_RF_dwt[train], data['label'][train]),
          accuracy_score(preds_RF_dwt[test], data['label'][test])))

