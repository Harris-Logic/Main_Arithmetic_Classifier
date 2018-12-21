# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 16:34:11 2018

@author: yan
SVC¶
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
work_dir = "A:\\cuckoo_json\\newcsv.csv"
bankdata = pd.read_csv(work_dir) 
#bankdata.shape 这他娘的是命令行用的代码
#bankdata.head()

#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()
#bankdata = bankdata.apply(le.fit_transform)

droplist = ['class']
X = bankdata.drop(droplist, axis=1)
y = bankdata['class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
#labels = np.unique(X); print(labels)

from sklearn.svm import SVC
clf = SVC(kernel='poly',degree=4)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#y_pred = svclassifier.predict(X_test)
#
#from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred)) 
#
#
#
#from sklearn.svm import SVC
#svclassifier = SVC(kernel='linear')
#svclassifier.fit(X_train, y_train)