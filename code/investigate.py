import os
import sys
import time

import numpy
from utils import  load_data_half

import cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

# w/ Ipython kernel
%matplotlib inline
dataset='mnist.pkl.gz'
datasets = load_data_half(dataset)

train_set_x, train_set_y = datasets[0]
test_set_x, test_set_y = datasets[1]
print train_set_x.get_value(borrow=True).shape


f = gzip.open('../data/'+dataset, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

X = train_set[0]
Y = train_set[1]
print X.shape
print Y.shape

print X[0]

image = Image.fromarray(np.uint8(cm.Greys_r(np.reshape(X[0],(28,28)))*255))
plt.imshow(image,cmap = cm.Greys_r)


data_npy = numpy.load('../data/mnist_pairs_da.npz')
train_set_x = data_npy['train_set_x']
train_set_y = data_npy['train_set_y']
test_set_x = data_npy['test_set_x']
test_set_y = data_npy['test_set_y']
print train_set_x[0,:]
