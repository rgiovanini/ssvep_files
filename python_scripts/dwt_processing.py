#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 19:26:50 2017

@author: renato

Getting and saving the Wavelet Decomposition arrays
"""
import numpy as np
import pywt 
import RF_estimator as rf
import cross_validation as cv

data = np.load("/home/renato/Dropbox/Mestrado/final/python_scripts/all_data_scaled.npy")
target = np.load("/home/renato/Dropbox/Mestrado/final/python_scripts/target.npy")

# decomposing using the Discrete Wavelet transform 
n_levels = 4

data_scaled_dwt = [pywt.wavedec(data[i], pywt.Wavelet('db8'),
                   level=n_levels)[0] for i in range(len(data))]
data_scaled_dwt = np.vstack(data_scaled_dwt)

#np.save('data_dwt_db12_level8.npy', data_scaled_dwt)

# cross validating and applying the Random Forest model 
n_trials = 20

clf, train_sc, test_sc, data_sets, target_sets = rf.random_forest_classifier(data=data_scaled_dwt,
                                                                             target=target,
                                                                             test_size=0.33, 
                                                                             n_trials=n_trials,
                                                                             n_estimators=100)

cv_score = cv.cross_validation(estimator=clf, data=data_sets, 
                            target=target_sets, n_trials=n_trials)

print("Results for n_trials = ", n_trials)
print("Decomposition level: ", n_levels)
print("Train accuracy: {:.2%} +/- {:.2%}".format(train_sc[0], train_sc[1]))
print("Test accuracy: {:.2%} +/- {:.2%}".format(test_sc[0], test_sc[1]))
print("cross validation score: {:.2%} +/- {:.2%}".format(cv_score[0], cv_score[1])) 
print("--------------------------------------------")
  