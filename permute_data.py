# coding: utf-8
# Author: Zhongyang Zhang

import h5py
import numpy as np
import pickle

root = '/Volumes/Takanashi/Datasets_Repo/POISSON/'#'./TempData/'
DATA_PATH = [root + 'train_data_2.mat', root + 'test_data_2.mat']
train_data = h5py.File(DATA_PATH[0], 'r')
test_data = h5py.File(DATA_PATH[1], 'r')

train_data = dict((key, value) for key, value in train_data.items() if key == 'X_2_train' or key == 'Y_train')
test_data = dict((key, value) for key, value in test_data.items() if key == 'X_2_test' or key == 'Y_test')
train_data_X = np.transpose(train_data['X_2_train'], (3, 2, 1, 0))
train_data_Y = np.transpose(train_data['Y_train'], (2, 1, 0))
test_data_X = np.transpose(test_data['X_2_test'], (3, 2, 1, 0))
test_data_Y = np.transpose(test_data['Y_test'], (2, 1, 0))

train_pairs = [(x, y.T.reshape(-1)) for x, y in zip(train_data_X, train_data_Y)]
test_pairs = [(x, y.T.reshape(-1)) for x, y in zip(test_data_X, test_data_Y)]

pickle.dump(train_pairs, open(root+'train_data_2.pkl', 'wb+'))
pickle.dump(test_pairs, open(root+'test_data_2.pkl', 'wb+'))

