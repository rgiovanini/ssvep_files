# Aplicação de cross validation no sistema de classificação
import numpy as np
from scipy.io import loadmat

# eeg = loadmat('/home/renato/Dropbox/Mestrado/final/'
# 			   'kolodziej_dataset/dataset_7_canais.mat')

# carregando os canais de interesse dos arquivos .mat e gerando o array
lista_canais = list(range(15))
n_canais = len(lista_canais)
path = '/home/renato/Dropbox/Mestrado/final/kolodziej_dataset/all_data/S{}_{}Hz.mat'
data = [np.transpose(loadmat(path.format(i, j))['X'])[lista_canais]
        for j in range(5, 9) for i in range(1, 6)]
data = np.vstack(data)

# criando o array de targets
target = [np.full(n_canais * 5, i, dtype='int64') for i in range(5, 9)]
target = np.hstack((target[0], target[1], target[2], target[3]))

# data = eeg['data']
# target = eeg['target']
# target = target.reshape(target.shape[1])

from sklearn.preprocessing import scale

data_scaled = np.array([scale(data[i]) for i in range(len(data))])

import pywt

# decompondo em sub-bandas usando a DWT e wavelet db12 e tomando
# apenas as aproximações
n_levels = 2
data_scaled_dwt = [pywt.wavedec(data_scaled[i], pywt.Wavelet('db12'),
                   level=n_levels)[0] for i in range(len(data_scaled))]
data_scaled_dwt = np.vstack(data_scaled_dwt)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

clf_RF = RandomForestClassifier(n_estimators=1000, max_features='auto',
                                    n_jobs=-1, bootstrap=True,
                                    warm_start=0, oob_score=True)

for i in range(3,10):
	CV_score = cross_val_score(clf_RF, data, target, cv=i, n_jobs=-1)
	print("Accuracy after Cross Validation: {:.2%} +/- {:.2%}".format(
		   CV_score.mean(), CV_score.std()))	