#-*-coding:utf-8-*-#

import numpy as np
import scipy.io as sio
from sklearn import svm

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

x = data[:,:150]
y = newlabel[:,:]

sio.savemat('label.mat',{'y':y})
sio.savemat('data.mat',{'x':x})

x_train_ = x[0:43680,:]
x_test_ = x[43680:53872,:]
y_train_ = y[0:43680, :]
y_test_ = y[43680:53872,:]


x_train = x_train_.reshape(x_train_.shape[0]/91,91,x_train_.shape[1])
x_test = x_test_.reshape(x_test_.shape[0]/91,91,x_test_.shape[1])

y_train = np.zeros((y_train_.shape[0]/91,y_train_.shape[1]),dtype =np.int)
y_test = np.zeros((y_test_.shape[0]/91,y_test_.shape[1]),dtype = np.int)

index1 = 0
index2 = 0
for i in range(0,43680,91):
    y_train[index1,:] = y_train_[i,:]
    index1 = index1 + 1

for j in range(0,10192,91):
    y_test[index2,:] = y_test_[j,:]
    index2 = index2 + 1

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

sio.savemat('x_train',{'x_train':x_train})
sio.savemat('y_train',{'y_train':y_train})
sio.savemat('x_test',{'x_test':x_test})
sio.savemat('y_test',{'y_test':y_test})






