# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:17:59 2018

@author: DELL
"""
import numpy as np
import time
import random
class SVM(object):
    def __init__(self, kernel = 'rbf', e = 0.001, p = 0, C = 10, maxiteration = 1000):        
    	self.kernel = kernel
    	self.p = p
    	self.e = e
    	self.C = C
    	self.maxiteration = maxiteration

    def _init_parameters(self, features, labels):
        self.X = features
        self.y = labels.reshape(-1,1)
        self.N = len(self.X)
        self.b = 0
        self.counter = 0
        #worth discussion start value
        self.a = np.zeros((self.N,1))

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
    # def W(self, list):
    # 	n = len(y)
    #     X = X.reshape(-1,n)
    #     y = y.reshape(n,-1)
    #     return np.dot(np.dot((self.a[list].T*self.y[list].T),
    #     	self.K(self.X[:,list],self.X[:,list])),self.a*self.y)/2

    # above one seems better.
    # def W(self, X, y):
    # 	n = len(y)
    #     X = X.reshape(-1,n)
    #     y = y.reshape(n,-1)
    #     return np.dot(np.dot((a.T*y.T),self.K(X,X)),a*y)/2

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
# the original idea which costs too much time.
#    def find2para(self, save = True):
#        index_list = [i for i in range(self.N)]
#        i1_list_1 = filter(lambda i: self.a[i] > 0 and self.a[i] < self.C, index_list)
#        i1_list_2 = list(set(index_list) - set(i1_list_1))
#        i1_list_1 = list(i1_list_1)
#    #change the sequence of the list, since we want 0<a<C to be tested first.
#        i1_list = i1_list_1
#        i1_list.append(i1_list_2)
#        for i in i1_list[0]:
#            if self.KKT(i):
#                continue
#    #potential optimization, find the max disobey a1.
#            E1 = self.E(i)
#            m = (0, 0)
#            if save == True:
#                for j in index_list:
#                    if i == j:
#                        continue
#                    E2 = self.E(j)
#                    if abs(E1 - E2) > m[0]:
#                        m = (abs(E1 - E2), j)       
#                return i, m[1]
#            else:
#                j = random.randint(0,self.N-1)               
#                while j==i:
#                    j = random.randint(0,self.N-1) 
#                print(i,j)
#                return i,j
#        return [0]

# a easier and quicker solution for choosing parameters.             
    def find2para(self, save = True):
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
    
    def adjust(self, i, j):
    	if self.y[i] != self.y[j]:
            L = max(0, self.a[j] - self.a[i])
            H = min(self.C, self.C + self.a[j]-self.a[i])
    	else:
            L = max(0, self.a[j] + self.a[i] - self.C)
            H = min(self.C, self.a[j] + self.a[i])
    	eta = self.K(self.X[i], self.X[i]) + self.K(self.X[j], self.X[j]) - 2*self.K(self.X[i], self.X[j]) 	
    	ajnu = self.a[j] + (self.y[j]*(self.E(i)-self.E(j)))/eta
    	if ajnu > H:
    		ajn = H
    	elif ajnu < L:
    		ajn = L
    	else:
    		ajn = ajnu
    	ain = self.a[i] + self.y[i] * self.y[j] * (self.a[j] - ajn)
        #E is not updated
    	if (ain<self.C) and (ain>0):
    		b = -self.E(i) - self.y[i] * self.K(self.X[i], self.X[i]) * (ain- self.a[i]) - self.y[j] * self.K(self.X[j], self.X[i]) * (ajn - self.a[j]) + self.b
    	elif (ajn<self.C) and (ajn>0):
    		b = -self.E(j) - self.y[i] * self.K(self.X[i], self.X[j]) * (ain- self.a[i]) - self.y[j] * self.K(self.X[j], self.X[j]) * (ajn - self.a[j]) + self.b
    	else:
    		b = (-self.E(i) - self.y[i] * self.K(self.X[i], self.X[i]) * (ain- self.a[i]) - self.y[j] * self.K(self.X[j], self.X[i]) * (ajn - self.a[j]) + self.b)/2 + \
        	(-self.E(j) - self.y[i] * self.K(self.X[i], self.X[j]) * (ain- self.a[i]) - self.y[j] * self.K(self.X[j], self.X[j]) * (ajn - self.a[j]) + self.b)/2   	        
    	return ain, ajn, b
    #here worth further discussion
        	

    def train(self, features, labels):
    	self._init_parameters(features, labels)
    	save = True; time = 0; l_one = -1; l_two = -1        
    	while self.counter < self.maxiteration:               
            if len(self.find2para(save)) == 1: break
            n_one, n_two = self.find2para(save)
            save = True
            self.counter += 1
            if (n_one == l_one) and (n_two == l_two):
                time += 1
            else: 
                l_one = n_one 
                l_two = n_two
            if time >=5: 
                save = False
                time = 0           
            self.a[l_one], self.a[l_two], self.b = self.adjust(l_one, l_two)              
    	print(self.a)
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
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=3)
#    X = np.append(([500,500],[0,0]),X).reshape(-1,2)
#    y = np.append([0,1],y)
    for i in range(len(y)):
    	if y[i] == 0:
            y[i] = -1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = np.append(([500,500],[0,0]),X_train).reshape(-1,2)
    y_train = np.append([-1,1],y_train)
    
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
    svm = SVM(kernel = 'rbf', e = 0.001, p = 0.2, C = 1, maxiteration = 1000)
    svm.train(X_train,y_train)
    print ('Start predicting')
    score = accuracy_score(svm.predict(X_test).T,y_test) 
    time_4 = time.time()
    print ("svmms the accruacy socre is ", score)
    print ('time cost ',time_4 - time_3,' second','\n') 

#X, y = make_gaussian_quantiles(cov=2,n_samples=100, n_features=2,n_classes=2, random_state=1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#svm the accruacy socre is  0.875
#time cost  0.0019991397857666016  second 
#
#svmms the accruacy socre is  0.916666666667  (rbf, 0.001, 0.1, 1, 164)
#time cost  3.3493804931640625  second 


#svm the accruacy socre is  0.955555555556
#time cost  0.0030019283294677734  second 

#svmms the accruacy socre is  0.955555555556 (rbf, 0.001, 0.1, 1, 283)
#time cost  8.59010648727417  second 



