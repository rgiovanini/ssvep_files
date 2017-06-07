# -*- coding: utf-8 -*-
"""
Spyder Editor

Processamento de sinais do dataset Kolodziej
"""

import numpy as np
from scipy.io import loadmat
#import matplotlib.pyplot as plt

data = loadmat('/media/renato/Dados/Mestrado/SSVEP-Datasets/Kolodziej/dataset_kolodziej.mat')

# modificando a key 'data' para deixar semelhante à do exemplo
data['data'] = np.vsplit(data['data'],data['data'].shape[0])
data['data'] = [np.reshape(data['data'][i],data['data'][i].shape[1]) 
                for i in range(len(data['data']))]
data['label'] = data['label'].reshape(data['label'].shape[1])

eeg_data = data['data']
target = data['label']#.reshape(320)

# whitening dos dados (testar acuracia com e sem isso)
from sklearn import preprocessing

eeg1 = preprocessing.scale(eeg_data)
eeg2 = np.array([preprocessing.scale(eeg_data[k]) 
                for k in range(len(eeg_data))])

eeg = [eeg_data, eeg1, eeg2]

# Tenho 3 vetores de instancias: sem whitening, com whitening pela media de 
# todo o array e das instancias separadas (eeg, eeg1, eeg2)
# gerando quais as features a serem analisadas
features_to_use = ["amplitude",
                   "percent_beyond_1_std",
                   "maximum",
                   "max_slope",
                   "median",
                   "median_absolute_deviation",
                   "percent_close_to_median",
                   "minimum",
                   "skew",
                   "std",
                   "weighted_average"]

#definindo novamente as features do Guo et al
from scipy.stats import skew
def mean_signal(t, m, e):
    return np.mean(m)

def std_signal(t, m, e):
    return np.std(m)

def mean_square_signal(t, m, e):
    return np.mean(m ** 2)

def abs_diffs_signal(t, m, e):
    return np.sum(np.abs(np.diff(m)))

def skew_signal(t, m, e):
    return skew(m)

guo_features = {
    'mean': mean_signal,
    'std': std_signal, 
    'mean2': mean_square_signal, 
    'abs_diffs': abs_diffs_signal, 
    'skew': skew_signal
}


# decomposição em sub-bandas utilizando a transformada Wavelet discreta
import pywt
from cesium import featurize

n_bands = 5
dwt_values=[]
fset_dwt=[]

dwt_values = [pywt.wavedec(m, pywt.Wavelet('db6'), level=n_bands-1)for m in eeg2]
fset_dwt = featurize.featurize_time_series(times=None, values=dwt_values, errors=None, features_to_use=features_to_use)

fset_guo_dwt = featurize.featurize_time_series(times=None, values=dwt_values, errors=None, 
                                               features_to_use=list(guo_features.keys()), 
                                               custom_functions=guo_features)


# tenho uma lista de dwts para os 3 casos (dwt_values) e uma lista com as respectivas features (fset_dwt)

# CLASSIFICAÇÃO 
    
# Random Forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train, test = train_test_split(np.arange(len(eeg[0])), test_size=0.4) #faço um split dos indices para train e test


model_RF_dwt = RandomForestClassifier(n_estimators=1800, max_features='auto', random_state=0, n_jobs=-1, 
                                      bootstrap=True, warm_start=0, oob_score=True)
    
model_RF_dwt.fit(fset_guo_dwt.iloc[train], target[train])
preds_RF_dwt = model_RF_dwt.predict(fset_guo_dwt)
    
print("Wavelet transform features using RF: training accuracy={:.2%}, test accuracy={:.2%}".format(
          accuracy_score(preds_RF_dwt[train], target[train]),
          accuracy_score(preds_RF_dwt[test], target[test])))