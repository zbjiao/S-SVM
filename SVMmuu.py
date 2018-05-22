# -*- coding: utf-8 -*-
"""
Created on Fri May 18 14:12:59 2018

@author: DELL
"""

import numpy as np
import time
import random
import math
class SVM(object):
    def __init__(self, kernel = 'rbf', e = 0.001, p = 0, C = 10, maxiteration = 1000, kesie = 1): 
        self.kernel = kernel
        self.kesie = kesie 
        self.p = p
        self.e = e
        self.C = C/2
        self.maxiteration = maxiteration


    def _init_parameters(self, features, labels):
        self.X = features
        self.y = labels.reshape(-1,1)
        self.N = len(self.X)
        self.b = 0
        self.counter = 0
        #worth discussion start value
        self.a = np.zeros((self.N,1))
        self.l = 0
        self.sum1 = 0
        self.sum2 = 0
        # as long as self.a starts with 0, sum1 and sum2 would start with 0.       

    def K(self, x, y):
        if self.kernel == 'poly':
             if self.p==0:
                 return "need a p"
             return (np.dot(x.T,y)+1)**self.p
        if self.kernel == 'rbf':
             if self.p == 0:
                 return "need a phi"
             ex = np.linalg.norm(x.reshape(-1,1)-y.reshape(-1,1), ord=2)
             return np.exp(-np.power(ex,2)/(2*np.power(self.p,2)))
        if self.kernel == 'linear':
            return np.dot(x.T,y)
        return "false kernel"
    
    def KK(self, x, y):
        if self.kernel == 'poly':
             if self.p<=0:
                 return "need a p"
             return (np.dot(x,y.T)+np.ones((x.shape[0],y.shape[0])))**self.p
        if self.kernel == 'rbf':
             if self.p == 0:
                 return "need a phi"
             sum = np.zeros(shape=(x.shape[0],y.shape[0]))
             for i in range(x.shape[0]):
                 for j in range(y.shape[0]):
                     ex = np.linalg.norm(x[i]-y[j], ord=2)
                     sum[i,j] = np.exp(-np.power(ex,2)/(2*np.power(self.p,2)))
             return sum 
        if self.kernel == 'linear':
            return np.dot(x,y.T)
        return "false kernel"

    def g(self, i):        
        return np.dot((self.a*self.y).T, self.KK(self.X, self.X[i].reshape(1,-1))) + self.b
    
    def E(self, i):
    	return self.g(i)-self.y[i]

    def KKT(self, i):
        temp = self.y[i]*self.g(i)
        if abs(self.a[i]) < self.e:
            return temp >= 1
        elif abs(self.C-self.a[i]) < self.e:
            return temp <= 1
        else:         
            return temp == 1
             
    def find2para(self):
        index_list = [i for i in range(self.N)]
        i1_list_1 = filter(lambda i: self.a[i] > 0 and self.a[i] < self.C, index_list)
        i1_list_2 = list(set(index_list) - set(i1_list_1))
        i1_list_1 = list(i1_list_1)
    #change the sequence of the list, since we want 0<a<C to be tested first.
        i1_list = i1_list_1
        i1_list.append(i1_list_2) 
        
        for i in i1_list[0]:
            if self.KKT(i):
                continue
            j = random.randint(0,self.N-1)               
            while j==i:
                j = random.randint(0,self.N-1) 
            return i,j
        return [0]

    # newton raphson is not applicable since there is no explicit solution for its dx.
    def findl(self):
        left = 0
        right = 1
        lm = 2/3*left + 1/3*right
        rm = 2/3*right + 1/3*left
        while (right-left)>0.0001:
            gp1 = self.gp(lm)
            gp2 = self.gp(rm)
            if gp1>gp2:
                right = rm
            else:
                left = lm
            lm = 2/3*left+1/3*right
            rm = 2/3*right+1/3*left
        return (lm+rm)/2            
        
    def gp(self, a):
        return ((1-a*a)*math.log((1-a)/(1+a)))/(-2*self.kesie) + a 
    
    def adjust(self, i, j):
        k11 = self.K(self.X[i], self.X[i])
        k22 = self.K(self.X[j], self.X[j])
        k12 = self.K(self.X[i], self.X[j])

        if self.y[i] != self.y[j]:
            L = max(0, self.a[j] - self.a[i])
            H = min(self.C, self.C + self.a[j]-self.a[i]) 

        else:
            L = max(0, self.a[j] + self.a[i] - self.C)
            H = min(self.C, self.a[j] + self.a[i])   
        
        #it seems that we dont have to calculate sum1 and sum2 everytime since only two terms of them have changed 

        self.sum1 = self.sum1 - self.y[i]*self.a[i]*k11 - self.y[j]*self.a[j]*k12
        self.sum2 = self.sum2 - self.y[i]*self.a[i]*k12 - self.y[j]*self.a[j]*k22     
        

        if H <= L:
            ain = self.a[i] + self.y[i]*self.y[j]*(self.a[j] - L)
            ajn = L
        else:
            lm = H/3+2*L/3
            rm = L/3+2*H/3
            left = L
            right = H
            while (right - left)>0.001:
                a2 = self.a[i] + self.y[i]*self.y[j]*(self.a[j] - lm)
                a3 = self.a[i] + self.y[i]*self.y[j]*(self.a[j] - rm)
                ww2 = k11*a2*a2/2 + k22*lm*lm/2 + self.y[i]*self.y[j]*k12*a2*lm - a2 - lm + self.y[i]*a2*self.sum1 + \
            self.y[j]*lm*self.sum2 - 2*self.C*self.gp(min(self.l,math.sqrt(max((1-a2/self.C),0)))) -2*self.C*self.gp(min(self.l,math.sqrt(max((1-lm/self.C),0))))
                ww3 = k11*a3*a3/2 + k22*rm*rm/2 + self.y[i]*self.y[j]*k12*a3*rm - a3 - rm + self.y[i]*a3*self.sum1 + \
            self.y[j]*rm*self.sum2 - 2*self.C*self.gp(min(self.l,math.sqrt(max((1-a3/self.C),0)))) -2*self.C*self.gp(min(self.l,math.sqrt(max((1-rm/self.C),0))))
                if ww2 < ww3:
                    right = rm
                else:
                    left = lm
                lm = right/3+2*left/3
                rm = left/3+2*right/3  
            ajn = (lm + rm)/2
            ain = self.a[i] + self.y[i]*self.y[j]*(self.a[j] - ajn)

        if (ain<self.C) and (ain>0):
            b = -self.E(i) - self.y[i] * k11 * (ain- self.a[i]) - self.y[j] * k12 * (ajn - self.a[j]) + self.b
        elif (ajn<self.C) and (ajn>0):
            b = -self.E(j) - self.y[i] * k12 * (ain- self.a[i]) - self.y[j] * k22 * (ajn - self.a[j]) + self.b
        else:
            b = (-self.E(i) - self.y[i] * k11 * (ain- self.a[i]) - self.y[j] * k12 * (ajn - self.a[j]) + self.b)/2 + \
        	(-self.E(j) - self.y[i] * k12 * (ain- self.a[i]) - self.y[j] * k22 * (ajn - self.a[j]) + self.b)/2
  	     
        self.sum1 = self.sum1 + self.y[i]*ain*k11 - self.y[j]*ajn*k12
        self.sum2 = self.sum2 + self.y[i]*ain*k12 - self.y[j]*ajn*k22   
        return ain, ajn, b
    #here worth further discussion
        	 
    def train(self, features, labels):
        self._init_parameters(features, labels)
        l_one = -1; l_two = -1  
        self.l = self.findl()
        
        while self.counter < self.maxiteration:             
            if len(self.find2para()) == 1: break
        
            l_one, l_two = self.find2para()
            self.counter += 1        
            self.a[l_one], self.a[l_two], self.b = self.adjust(l_one, l_two)            
        
        print(self.counter)


    def predict(self, samples):
        answer = np.sign(np.dot((self.a*self.y).T, self.KK(self.X, samples)+self.b))
        return answer


if __name__ == "__main__":
    
    from sklearn.datasets import make_gaussian_quantiles 
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    print ('Start read data')
    X, y = make_gaussian_quantiles(cov=2,
                                 n_samples=1000, n_features=2,
                                 n_classes=2, random_state=3)
    for i in range(len(y)):
    	if y[i] == 0:
            y[i] = -1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = np.append(([500,500],[0,0]),X_train).reshape(-1,2)
    y_train = np.append([-1,1],y_train)

#########################C-SVM#################################        
    print ('Start training')
    time_1 = time.time()
    
    clf = SVC(kernel = 'rbf')
    clf.fit(X_train,y_train)
    
    print ('Start predicting')
    score = accuracy_score(clf.predict(X_test),y_test)   
    time_2 = time.time()
    print ("svm the accruacy socre is ", score)
    print ('time cost ',time_2 - time_1,' second','\n')
    
#######################OUTLIER-INSENSITIVE SVM#################
    print ('Start training')
    time_3 = time.time()    
    svm = SVM(kernel = 'rbf', e = 0.001, p = 0.2, C = 0.03, maxiteration = 1000, kesie = 2)
    svm.train(X_train,y_train)
    print ('Start predicting')
    score = accuracy_score(svm.predict(X_test).T,y_test) 
    time_4 = time.time()
    print ("svmms the accruacy socre is ", score)
    print ('time cost ',time_4 - time_3,' second','\n') 