# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:54:25 2018

@author: DELL
"""

import pandas as pd
import SVMmu
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#wisconsin breast cancer
df = pd.read_csv('wdbc.data')
df = df.iloc[:,1:]
y = df.iloc[:,0]
X = df.iloc[:,1:]
y = y.map({'M':1, "B":-1})
X = X.values
##transfusion.data
#df = pd.read_csv('transfusion.data')
#X = df.iloc[:,:-1]
#y = df.iloc[:,-1]
#y = y.map({1:1,0:-1})
#X = X.values

#####transfusion-result#####
#S-SVM
#687
#Start predicting
#svmms the accruacy socre is  0.768888888889
#time cost  357.1682319641113  second 
#
#C-SVM
#svm the accruacy socre is  0.733333333333
#time cost  0.02701711654663086  second 
################



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

from sklearn.svm import SVC
print ('Start training')
time_1 = time.time()

clf = SVC(kernel = 'rbf')
clf.fit(X_train,y_train)

print ('Start predicting')
score = accuracy_score(clf.predict(X_test),y_test)   
time_2 = time.time()
print ("svm the accruacy socre is ", score)
print ('time cost ',time_2 - time_1,' second','\n')

time_3 = time.time()
print ('Start training')
svm = SVMmu.SVM(kernel = 'rbf', e = 0.001, p = 0.5, C = 1, maxiteration = 1000, kesie = 1)
svm.train(X_train,y_train)
print ('Start predicting')
score = accuracy_score(svm.predict(X_test).T,y_test) 
time_4 = time.time()
print ("svmms the accruacy socre is ", score)
print ('time cost ',time_4 - time_3,' second','\n') 