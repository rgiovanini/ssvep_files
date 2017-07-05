#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:13:35 2017

@author: renato

Function to apply Random Forest Estimator
"""
def random_forest_classifier(data, target, test_size=0.5, n_estimators=500, n_trials=10):
    '''
    FUNCTION DESCRIPTION 
    '''
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    for i in range(n_trials):
        # creating index vector to split data into train and test sets
        train, test = train_test_split(np.arange(len(target)), test_size=test_size)
        
        # creating the estimator
        clf_RF = RandomForestClassifier(n_estimators=n_estimators, 
                                        max_features='auto',
                                        n_jobs=-1,
                                        bootstrap=True,
                                        warm_start=0,
                                        oob_score=True)
           
        # fitting the estimator with the data 
        clf_RF.fit(data[train], target[train])
        
        # predicting and accuracy score 
        train_list, test_list = [], []
        
        pred_train = clf_RF.predict(data[train])
        pred_test = clf_RF.predict(data[test])
        
        train_list.append(accuracy_score(pred_train, target[train]))
        test_list.append(accuracy_score(pred_test, target[test]))
        
        train_scores = (np.mean(train_list), np.std(train_list))
        test_scores = (np.mean(test_list), np.std(test_list))
        
        data_to_cv = data[train]
        target_to_cv = target[train]

    return clf_RF, train_scores, test_scores, data_to_cv, target_to_cv


if __name__ == "__main__":
    
    import numpy as np
    import cross_validation as cv
    
    data = np.load("/home/renato/Dropbox/Mestrado/final/python_scripts/all_filtered_signal_dec100.npy")
    target = np.load("/home/renato/Dropbox/Mestrado/final/python_scripts/target.npy")
    
    # some usage example 
    n_trials = 20
    for i in range(4):  # para cada banco de filtros
        clf, train_sc, test_sc, data_sets, target_sets = random_forest_classifier(data=data[i],
                                                                                  target=target,
                                                                                  test_size=0.33, 
                                                                                  n_trials=n_trials,
                                                                                  n_estimators=100)
        
        cv_score = cv.cross_validationcross_validation(estimator=clf, data=data_sets, 
                                    target=target_sets, n_trials=n_trials)
        
        print("Results for n_trials = ", n_trials)
        print("Train accuracy: {:.2%} +/- {:.2%}".format(train_sc[0], train_sc[1]))
        print("Test accuracy: {:.2%} +/- {:.2%}".format(test_sc[0], test_sc[1]))
        print("cross validation score: {:.2%} +/- {:.2%}".format(cv_score[0], cv_score[1])) 
        print("--------------------------------------------")