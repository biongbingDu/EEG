#-*-coding:utf-8-*-#


import numpy as np
import scipy.io as sio
import math
import h5py

#load data
datamat = h5py.File('./DEAPdata/data.mat')
data = np.transpose(datamat['dataAll'])

labelsmat = h5py.File('./DEAPdata/labels.mat')
labels = np.transpose(labelsmat['labelsAll'])

print('load data done!')

#transform data
dataShape = data.shape
Tdata = np.zeros((dataShape[0],dataShape[1],108),dtype=np.int)
labelsShape = labels.shape
Tlabels = np.zeros((labelsShape[0],1),dtype=np.int)

print('transform data done!')

#set labels
for i in range (0,labelsShape[0]):
    if labels[i,0] >= 5 and labels[i,1] >= 5:
        Tlabels[i,0] = 3
    if labels[i,0] >= 5 and labels[i,1] < 5:
        Tlabels[i,0] = 2
    if labels[i,0] < 5 and labels[i,1] >= 5:
        Tlabels[i,0] = 1
    if labels[i,0] < 5 and labels[i,1] < 5:
        Tlabels[i,0] = 0

y = Tlabels

print('set labels done!')

#skew-kur
def calc(data):
    n = len(data)
    niu = 0.0
    niu2 = 0.0
    niu3 = 0.0
    for a in data:
        niu += a
        niu2 += a**2
        niu3 += a**3
    niu/= n   #这是求E(X)
    niu2 /= n #这是E(X^2)
    niu3 /= n #这是E(X^3)
    sigma = math.sqrt(niu2 - niu*niu) #这是D（X）的开方，标准差
    return [niu,sigma,niu3] #返回[E（X）,标准差，E（X^3）]

def calc_stat(data):
    [niu,sigma,niu3] = calc(data)
    n = len(data)
    niu4 = 0.0
    for a in data:
        a -= niu
        niu4 += a ** 4
    niu4 /= n
    skew = (niu3 - 3*niu*sigma**2 - niu**3)/(sigma**3)
    kurt =  niu4/(sigma**2)

    if not skew == skew:
        skew = 0
    if not kurt == kurt:
        kurt = 0

    return [niu,sigma,skew,kurt] #返回了均值，标准差，偏度，峰度

print('skew-kur done!')

#Trans data
for F in range(0,1280):
    for S in range(0,40):
        index = 0

        print('Trans data ................................')

        for j in range (0,8060,806):
            h = j+806
            mean = np.mean(data[F,S,j:h],axis = 0)
            median = np.median(data[F,S,j:h],axis = 0)
            maximum = np.max(data[F,S,j:h],axis = 0)
            minimum = np.min(data[F,S,j:h],axis = 0)
            std = np.std(data[F,S,j:h],axis = 0,ddof = 1)
            variance = np.var(data[F,S,j:h],axis = 0,ddof = 1)
            ran = maximum - minimum
            skew = calc_stat(data[F,S,j:h])[2]
            kurt = calc_stat(data[F,S,j:h])[3]

            print('skew')
            print(skew)
            print('kurt')
            print(kurt)

            #the last 4 data
            if h == 8060:
                Tdata[F, S, index] = mean
                Tdata[F, S, index + 1] = median
                Tdata[F, S, index + 2] = maximum
                Tdata[F, S, index + 3] = minimum
                Tdata[F, S, index + 4] = std
                Tdata[F, S, index + 5] = variance
                Tdata[F, S, index + 6] = ran
                Tdata[F, S, index + 7] = skew
                Tdata[F, S, index + 8] = kurt
                index = index + 9;
                l = 8060
                t = l + 4;
                mean = np.mean(data[F, S, l:t], axis=0)
                median = np.median(data[F, S, l:t], axis=0)
                maximum = np.max(data[F, S, l:t], axis=0)
                minimum = np.min(data[F, S, l:t], axis=0)
                std = np.std(data[F, S, l:t], axis=0, ddof=1)
                variance = np.var(data[F, S, l:t], axis=0, ddof=1)
                ran = maximum - minimum
                skew = calc_stat(data[F, S, l:t])[2]
                kurt = calc_stat(data[F, S, l:t])[3]

                if t == 8064:
                    Tdata[F, S, index] = mean
                    Tdata[F, S, index + 1] = median
                    Tdata[F, S, index + 2] = maximum
                    Tdata[F, S, index + 3] = minimum
                    Tdata[F, S, index + 4] = std
                    Tdata[F, S, index + 5] = variance
                    Tdata[F, S, index + 6] = ran
                    Tdata[F, S, index + 7] = skew
                    Tdata[F, S, index + 8] = kurt
                    index = index + 9;
                    mean = np.mean(data[F, S, 0:t], axis=0)
                    median = np.median(data[F, S, 0:t], axis=0)
                    maximum = np.max(data[F, S, 0:t], axis=0)
                    minimum = np.min(data[F, S, 0:t], axis=0)
                    std = np.std(data[F, S, 0:t], axis=0, ddof=1)
                    variance = np.var(data[F, S, 0:t], axis=0, ddof=1)
                    ran = maximum - minimum
                    skew = calc_stat(data[F, S, 0:t])[2]
                    kurt = calc_stat(data[F, S, 0:t])[3]

                print('h =')
                print(h)
                print('l =')
                print(l)
                print('t =')
                print(t)

            #set Tdata
            Tdata[F, S, index] = mean
            Tdata[F, S, index + 1] = median
            Tdata[F, S, index + 2] = maximum
            Tdata[F, S, index + 3] = minimum
            Tdata[F, S, index + 4] = std
            Tdata[F, S, index + 5] = variance
            Tdata[F, S, index + 6] = ran
            Tdata[F, S, index + 7] = skew
            Tdata[F, S, index + 8] = kurt
            index = index + 9;

            print('F = ')
            print(F)
            print('S = ')
            print(S)
            print('j = ')
            print(j)





x = Tdata.transpose(0,2,1)

print('trans data done!')

x_train = x[0:1080,:,:]
x_test = x[1080:1280,:,:]

y_train = y[0:1080,:]
y_test = y[1080:1280,:]

sio.savemat('./DEAPdata/x',{'x':x})
sio.savemat('./DEAPdata/y',{'y':y})
sio.savemat('./DEAPdata/train/x_train',{'x_train':x_train})
sio.savemat('./DEAPdata/train/y_train',{'y_train':y_train})
sio.savemat('./DEAPdata/test/x_test',{'x_test':x_test})
sio.savemat('./DEAPdata/test/y_test',{'y_test':y_test})
print('save data done!')