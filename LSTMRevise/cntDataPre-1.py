#-*-coding:utf-8-*-#

import numpy as np
import scipy.io as sio
import math
import h5py

datamat = sio.loadmat('data_bmovie2.mat')
data = datamat['xlsdata']

dataShape = data.shape

datarow = dataShape[0]

# create label
newlabel = np.zeros((datarow, 1), dtype=np.int)
for i in range(0,datarow):
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

print('data extracting.................')

#extract features (delta)
Tx = np.random.random ((x.shape[0],x.shape[1]/5))

for f in range (0,x.shape[0]):
        index = 0
        for i in range (0,150,5):
            Tx[f,index] = x[f,i]
            index = index + 1
            print('index = ')
            print(index)

x = Tx
print('x.shape')
print(x.shape)

print('extract done!')

sio.savemat('./CNTdata-Pre-1/label.mat',{'y':y})
sio.savemat('./CNTdata-Pre-1/data.mat',{'x':x})

x_train_ = x[0:45500,:]
x_test_ = x[45500:53872,:]
y_train_ = y[0:45500, :]
y_test_ = y[45500:53872,:]


x_train = x_train_.reshape(x_train_.shape[0]/91,91,x_train_.shape[1])
x_test = x_test_.reshape(x_test_.shape[0]/91,91,x_test_.shape[1])

y_train = np.zeros((y_train_.shape[0]/91,y_train_.shape[1]),dtype =np.int)
y_test = np.zeros((y_test_.shape[0]/91,y_test_.shape[1]),dtype = np.int)

index1 = 0
index2 = 0
for i in range(0,45500,91):
    y_train[index1,:] = y_train_[i,:]
    index1 = index1 + 1

for j in range(0,8372,91):
    y_test[index2,:] = y_test_[j,:]
    index2 = index2 + 1

print('x_train_shape = ')
print(x_train.shape)
print('x_test_shape = ')
print(x_test.shape)
print('y_train_shape = ')
print(y_train.shape)
print('y_test_shape = ')
print(y_test.shape)

sio.savemat('./CNTdata-Pre-1/train/x_train',{'x_train':x_train})
sio.savemat('./CNTdata-Pre-1/train/y_train',{'y_train':y_train})
sio.savemat('./CNTdata-Pre-1/test/x_test',{'x_test':x_test})
sio.savemat('./CNTdata-Pre-1/test/y_test',{'y_test':y_test})


