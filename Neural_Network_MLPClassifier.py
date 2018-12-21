# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 19:29:11 2018

@author: yan
MLPClassifier¶
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

work_dir = "A:\\cuckoo_json\\newcsv.csv"
bankdata = pd.read_csv(work_dir) 

droplist = ['class']
X = bankdata.drop(droplist, axis=1)
y = bankdata['class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier()  #activation='logistic',solver='sgd',learning_rate='constant',learning_rate_init=0.001
                    
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))