
from __future__ import division
import numpy as np
import itertools
import random

"""
Gather traning and test sets for only similar pairs of digits
"""
# TODO Include in utils instead

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(xrange(n), r))
    return tuple(pool[i] for i in indices)

def get_pairs(labels, set):
    set_x = np.empty([0,28*28])
    set_y = np.empty([0,28*28])
    pairs_labels= []

    for digit in range(10):
        count = 0
        index = np.where(labels==digit)[0]
        print 'Digit ', digit, '#',len(index)
        for _ in range(1000):
            i,j = random_combination(range(len(index)), 2)
            I = index[i]
            J = index[j]
            assert(labels[I]==labels[J])
            set_x = np.vstack([set_x,np.expand_dims(set[I,:], axis=0)])
            set_y = np.vstack([set_y,np.expand_dims(set[J,:], axis=0)])
            pairs_labels.append(labels[I])
            count+=1
        print 'Gathered ',count,' pairs of ', digit

    return set_x, set_y, pairs_labels


if __name__ == '__main__':
    n = 10000
    # Training set:
    LABEL_FILE = '../data/t10k-labels-idx1-ubyte'
    with open(LABEL_FILE, 'rb') as f:
        f.read(8) # skip the header
        labels = np.fromstring(f.read(n), dtype=np.uint8)

    DATA_FILE = '../data/train-images-idx3-ubyte'
    with open(DATA_FILE, 'rb') as f:
        f.read(16) # skip the header
        set = np.fromstring(f.read(n * 28*28), dtype=np.uint8)
    set = set.reshape((n, 28*28))/255
    train_set_x, train_set_y, train_labels = get_pairs(labels, set)

    # Test set
    LABEL_FILE = '../data/t10k-labels-idx1-ubyte'
    with open(LABEL_FILE, 'rb') as f:
        f.read(8) # skip the header
        labels = np.fromstring(f.read(n), dtype=np.uint8)

    DATA_FILE = '../data/t10k-images-idx3-ubyte'
    with open(DATA_FILE, 'rb') as f:
        f.read(16) # skip the header
        set = np.fromstring(f.read(n * 28*28), dtype=np.uint8)
    set = set.reshape((n, 28*28))/255
    test_set_x, test_set_y, test_labels = get_pairs(labels, set)

    print 'Saving data to npz file...'
    np.savez('../data/mnist_pairs_da',train_set_x=train_set_x, train_set_y= train_set_y, train_labels = train_labels,\
        test_set_x = test_set_x, test_set_y = test_set_y, test_labels = test_labels)
    print 'Done'
