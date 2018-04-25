#-*-coding:utf-8-*-#

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm,datasets
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

#return accuracy
#def classification_report(pre_label,y_label):


#load dataset
datamat = sio.loadmat('data_bmovie2.mat')
data = datamat['xlsdata']

dataShape = data.shape
datarow = dataShape[0]

# create label
newlabel = np.zeros((datarow, 1), dtype=np.int)

for i in range(datarow):
    if data[i, 151:152] == 101:
        newlabel[i, 0] = 0;
    if data[i, 151:152] == 102 or data[i, 151:152] == 103 or data[i, 151:152] == 104:
        newlabel[i, 0] = 1;
    if data[i, 151:152] == 105 or data[i, 151:152] == 106 or data[i, 151:152] == 107:
        newlabel[i, 0] = 2;
    if data[i, 151:152] == 108 or data[i, 151:152] == 109:
        newlabel[i, 0] = 3
    if data[i, 151:152] == 110:
        newlabel[i, 0] = 4;
    if data[i, 151:152] == 111 or data[i, 151:152] == 112:
        newlabel[i, 0] = 5;
    if data[i, 151:152] == 113 or data[i, 151:152] == 114:
        newlabel[i, 0] = 6;
    if data[i, 151:152] == 115 or data[i, 151:152] == 116:
        newlabel[i, 0] = 7;

newlabelshape = newlabel.shape

#print(newlabelshape)
#print(newlabel)

# SVC()
clf = svm.SVC()

x = data[:,:150]
y = newlabel[:,:]

min_max_scaler = preprocessing.MinMaxScaler()
x_min_max = min_max_scaler.fit_transform(x)
x = x_min_max

sio.savemat('dataMinMax.mat',{'x':x})

#fit() training
x_train = x[0:43680,0:150]
y_train = y[0:43680,:]
clf.fit(x_train, np.ravel(y_train,order='C'))
print('training score')
print (clf.score(x_train,y_train))

#testing
x_test = x[43680:,:150]
y_test = y[43680:,:]

y_pre = clf.predict(x_test)
print('y_pre')
print(y_pre)
print('testing score')
print(clf.score(x_test,y_test))

# acc1 = accuracy_score(y_test,y_pre)
# print('acc1')
# print(acc1)

#predict() prediction
# pre_y = clf.predict(x[0:43680,:150])
# print(pre_y)
# print(y[0:43680])
# acc1 = accuracy_score(y[0:43680],pre_y)
# print(acc1)

# # prediction for test
# test_y = clf.predict(test)
# print (clf.score(test,test_y))
# print(test_y)
# acc2 = accuracy_score(y[43680:],test_y)
# print(acc2)

