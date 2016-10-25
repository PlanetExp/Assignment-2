# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:00:02 2016

@author: frederic

Script to read the forest dataset.

data[i] is the ith example, its class label is target[i]


"""
import pickle
import numpy as np

with open('forest_data.pickle', 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    data = pickle.load(f)
    target = pickle.load(f)
    
print(data.shape, data.dtype)
print(target.shape, target.dtype)
print data
print np.isnan(np.sum(data))

# from sklearn.cross_validation import train_test_split
# from sklearn import  preprocessing
#
# X_train,X_test,y_train,y_test = train_test_split(data,target, test_size=0.15, random_state=33)
# print X_train.shape, y_train.shape
#
#
# # Standardilize the features
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
#
# import  matplotlib.pyplot as plt
# color = ['red','greenyellow','blue']
# for i in xrange(len(color)):
#     xs = X_train[:,0][y_train == i]
#     ys = X_train[:,1][y_train == i]
#     plt.scatter(xs,ys,c=color[i])
# print target.dtype.names
# #plt.legend(target.dtype.names)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.show()


import numpy as np
# Substitute missing value with mean in the training data
# or with most common value
# or with median value
missing = data[:,1]
mean_missing = np.mean(data[missing !='NA',1].astype(np.float))
data[data[:,1] == 'NA',1] = mean_missing