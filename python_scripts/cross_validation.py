#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:17:29 2017

@author: renato

Cross Validation function
"""

def cross_validation(estimator, data, target, n_folds=6, n_trials=10):
    '''
    FUNCTION DESCRIPTION
    '''
    import numpy as np
    from sklearn.model_selection import cross_val_score
    
    cv_list = []
    for i in range(n_trials):
       cv_score = cross_val_score(estimator, data, target, cv=6, n_jobs=-1)
       cv_list.append(cv_score)
       
    return (np.mean(cv_list), np.std(cv_list))
