#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 12:18:11 2017

@author: renato
Evaluation of processing time for Random Forest method
"""

from numpy import load
from sklearn.externals import joblib
#from funcoes_filtragem import apply_fir_filter
# import one data array (one second, by example)

#test = load('teste.npy')
test = load('data_dwt_db8_level4_16ch.npy')

# import filter coefficients 
#coeff = load('timing/5hz_coefficients.npy')

# import estimator
#clf = joblib.load('RF_estimator_object.pkl')
clf_dwt= joblib.load('rf_estimator_object_dwt8_level4.pkl')

#applying filter to signal
#filtered_signal = apply_fir_filter(coeff, test)

#output = clf.predict(test)
output = clf_dwt.predict(test)