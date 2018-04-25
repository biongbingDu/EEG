# HAR classification 
# Author: Burak Himmetoglu
# 8/15/2017

import pandas as pd 
import numpy as np
import os
import scipy.io as sio

def read_data(data_path, split = "train"):
	""" Read data """

	# Fixed params
	n_class = 8
	n_steps = 91

	# Paths
	path_ = os.path.join(data_path, split)

	# Read labels and one-hot encode
	label_path = os.path.join(path_, "y_" + split + ".mat")
	labels_ = sio.loadmat(label_path)
	fileName2 = "y_"+split
	labels = labels_[fileName2]

	n_channels = 30

	# Initiate array
	X = np.zeros((n_steps, n_channels))
	dat_ = sio.loadmat(os.path.join(path_,"x_" + split +".mat"))
	fileName2 = "x_"+split
	data = dat_[fileName2]
	X = data

	return X, labels

def standardize(train, test):
	""" Standardize data """

	# Standardize train and test
	X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
	X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]

	return X_train, X_test

def one_hot(labels, n_class = 8):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y_ = expansion[:, labels-1].T
	y  = np.zeros((y_.shape[0],y_.shape[1]),dtype = np.int)
	y = y_[0,:,:]
	assert y.shape[1] == n_class, "Wrong number of labels!"

	return y



def get_batches(X, y, batch_size = 100):
	""" Return a generator for batches """
	n_batches = len(X) // batch_size
	X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

	# Loop over batches and yield
	for b in range(0, len(X), batch_size):
		yield X[b:b+batch_size], y[b:b+batch_size]


	




