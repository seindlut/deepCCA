from __future__ import division
from numpy.linalg import lstsq,eig
from numpy import cov,dot,arange,c_
import numpy as np
import leveldb
import itertools
import random
import caffe
def cca(x_tn,y_tm, reg=0.00000001):
    # Centering
    x_tn = x_tn-x_tn.mean(axis=0)
    y_tm = y_tm-y_tm.mean(axis=0)
    # Space dimension
    N = x_tn.shape[1]
    M = y_tm.shape[1]
    # Concatenate and evluate the empiric covar matrix
    xy_tq = c_[x_tn,y_tm]
    cqq = cov(xy_tq,rowvar=0)
    # Covar matrices with regularization
    cxx = cqq[:N,:N]+reg*np.eye(N)+0.000000001*np.ones((N,N))
    cxy = cqq[:N,N:(N+M)]+0.000000001*np.ones((N,M))
    cyx = cqq[N:(N+M),:N]+0.000000001*np.ones((M,N))
    cyy = cqq[N:(N+M),N:(N+M)]+reg*np.eye(M)+0.000000001*np.ones((M,M))

    K = min(N,M)

    xldivy = lstsq(cxx,cxy)[0]
    yldivx = lstsq(cyy,cyx)[0]

    _,vecs = eig(dot(xldivy,yldivx))
    a_nk = vecs[:,:K]
    b_mk = dot(yldivx,a_nk)

    u_tk = dot(x_tn,a_nk)
    v_tk = dot(y_tm,b_mk)

    return np.real(a_nk),np.real(b_mk),np.real(u_tk),np.real(v_tk)

def normr(a):
    return a/np.sqrt((a**2).sum(axis=1))[:,None]

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(xrange(n), r))
    return tuple(pool[i] for i in indices)

def get_pairs(labels, set):
    set_x_sim = np.empty([0,28*28])
    set_y_sim = np.empty([0,28*28])
    pairs_sim_labels= []

    for digit in range(10):
        count = 0
        index = np.where(labels==digit)[0]
        print 'Digit ', digit, '#',len(index)
        for _ in range(200):
            i,j = random_combination(range(len(index)), 2)
            I = index[i]
            J = index[j]
            assert(labels[I]==labels[J])
            set_x_sim = np.vstack([set_x_sim,np.expand_dims(set[I,:], axis=0)])
            set_y_sim = np.vstack([set_y_sim,np.expand_dims(set[J,:], axis=0)])
            pairs_sim_labels.append((digit, digit))
            count+=1
        print 'Gathered ',count,' pairs of ', digit
    # Add some dissimilar pairs:
    set_x_dissim = np.empty([0,28*28])
    set_y_dissim = np.empty([0,28*28])
    pairs_dissim_labels = []
    for d1 in range(10):
        count = 0
        index1 = np.where(labels==d1)[0]
        I = range(10)
        I.remove(d1)
        for d2 in I:
            index2 = np.where(labels==d2)[0]
            for _ in range(22):
                i= random.choice(index1)
                j= random.choice(index2)
                set_x_dissim = np.vstack([set_x_dissim,np.expand_dims(set[i,:], axis=0)])
                set_y_dissim = np.vstack([set_y_dissim,np.expand_dims(set[j,:], axis=0)])
                pairs_dissim_labels.append((d1, d2))
                count +=1
        print 'Gathered ',count,' pairs of dissimilars to ', d1
    return set_x_sim, set_y_sim, set_x_dissim, set_y_dissim, pairs_sim_labels, pairs_dissim_labels


def save_pairs():
    n = 10000
    # ------------------------------------------------------------------ Training set:
    LABEL_FILE = '../data/t10k-labels-idx1-ubyte'
    with open(LABEL_FILE, 'rb') as f:
        f.read(8) # skip the header
        labels = np.fromstring(f.read(n), dtype=np.uint8)

    DATA_FILE = '../data/train-images-idx3-ubyte'
    with open(DATA_FILE, 'rb') as f:
        f.read(16) # skip the header
        set = np.fromstring(f.read(n * 28*28), dtype=np.uint8)
    set = set.reshape((n, 28*28))/255
    train_x_sim, train_y_sim, train_x_dissim, train_y_dissim, train_sim_labels, train_dissim_labels = get_pairs(labels, set)

    # ------------------------------------------------------------------ Test set:
    LABEL_FILE = '../data/t10k-labels-idx1-ubyte'
    with open(LABEL_FILE, 'rb') as f:
        f.read(8) # skip the header
        labels = np.fromstring(f.read(n), dtype=np.uint8)

    DATA_FILE = '../data/t10k-images-idx3-ubyte'
    with open(DATA_FILE, 'rb') as f:
        f.read(16) # skip the header
        set = np.fromstring(f.read(n * 28*28), dtype=np.uint8)
    set = set.reshape((n, 28*28))/255
    test_x_sim, test_y_sim, test_x_dissim, test_y_dissim, test_sim_labels, test_dissim_labels = get_pairs(labels, set)


    print 'Saving data to npz file...'
    np.savez('../data/mnist_pairs_da',
    train_x_sim=train_x_sim,
    train_y_sim= train_y_sim,
    train_x_dissim=train_x_dissim,
    train_y_dissim= train_y_dissim,
    train_sim_labels = train_sim_labels,
    train_dissim_labels = train_dissim_labels,
    test_x_sim=test_x_sim,
    test_y_sim= test_y_sim,
    test_x_dissim=test_x_dissim,
    test_y_dissim= test_y_dissim,
    test_sim_labels = test_sim_labels,
    test_dissim_labels = test_dissim_labels)

    print '... done'


def write_leveldbs(X,Y, SIM, name=''):
    leveldb.DestroyDB('../data/'+name)
    db = leveldb.LevelDB('../data/'+name, create_if_missing=True, error_if_exists=True, write_buffer_size=268435456)
    wb = leveldb.WriteBatch()
    num_items = X.shape[0]
    print '# samples : ', num_items
    X = X.reshape(num_items, 28, 28)
    Y = X.reshape(num_items, 28, 28)
    count = 0
    for i in range(num_items):
        image1 = np.expand_dims(X[i,:,:], axis=0)
        image2 = np.expand_dims(Y[i,:,:], axis=0)
        image = np.vstack([image1, image2])
        sim = SIM[i]
        # Load image into datum object
        datum = caffe.io.array_to_datum(image, sim)
        wb.Put('%08d_%s' % (count, file), datum.SerializeToString())
        count = count + 1
        if count % 1000 == 0:
            # Write batch of images to database
            db.Write(wb)
            del wb
            wb = leveldb.WriteBatch()
            print 'Processed %i images.' % count

    if count % 1000 != 0:
        # Write last batch of images
        db.Write(wb)
        print 'Processed a total of %i images.' % count
    else:
        print 'Processed a total of %i images.' % count

def test_cca():
    # inputs :
    print '...loading data'
    # Loading the numpy arrays
    data_npy = np.load('../data/mnist_pairs_da.npz')
    train_x_sim=data_npy['train_x_sim']
    train_y_sim= data_npy['train_y_sim']
    train_x_dissim=data_npy['train_x_dissim']
    train_y_dissim= data_npy['train_y_dissim']
    train_sim_labels = data_npy['train_sim_labels']
    train_dissim_labels = data_npy['train_dissim_labels']
    test_x_sim=data_npy['test_x_sim']
    test_y_sim= data_npy['test_y_sim']
    test_x_dissim=data_npy['test_x_dissim']
    test_y_dissim= data_npy['test_y_dissim']
    test_sim_labels = data_npy['test_sim_labels']
    test_dissim_labels = data_npy['test_dissim_labels']

    print '...',len(train_x_sim), ' pairs'
    a,b,X_sim,Y_sim = cca(train_x_sim,train_y_sim)
    print 'a: ', a.shape
    print 'X: ', X_sim.shape
    # ---------------------------------------------- Train
    X_dissim = np.dot(train_x_dissim,a)
    Y_dissim = np.dot(train_y_dissim,b)

    # Merge and shuffle:
    X = np.vstack([X_sim, X_dissim])
    print 'X:',X.shape
    Y = np.vstack([Y_sim, Y_dissim])
    SIM = np.hstack([np.ones(len(X_sim),dtype=np.int), np.zeros(len(X_dissim),dtype=np.int)])
    pair_labels = np.vstack([train_sim_labels, train_dissim_labels])
    I = np.arange(X.shape[0])
    np.random.shuffle(I)
    X = X[I,:]
    Y = Y[I,:]
    SIM = SIM[I]
    pair_labels = pair_labels[I,:]
    np.savetxt('../data/listing_train.txt', pair_labels, fmt='%d')
    np.savetxt('../data/sim_train.txt', SIM, fmt='%d')
    write_leveldbs(X,Y, SIM, name='train')

    # ---------------------------------------------- Test
    X_sim = np.dot(test_x_sim,a)
    Y_sim = np.dot(test_y_sim,b)
    X_dissim = np.dot(test_x_dissim,a)
    Y_dissim = np.dot(test_y_dissim,b)

    # Merge and shuffle:
    X = np.vstack([X_sim, X_dissim])
    Y = np.vstack([Y_sim, Y_dissim])
    SIM = np.hstack([np.ones(len(X_sim),dtype=np.int), np.zeros(len(X_dissim),dtype=np.int)])
    pair_labels = np.vstack([test_sim_labels, test_dissim_labels])
    I = np.arange(X.shape[0])
    np.random.shuffle(I)
    X = X[I,:]
    Y = Y[I,:]
    SIM = SIM[I]
    pair_labels = pair_labels[I,:]
    np.savetxt('../data/listing_test.txt', pair_labels, fmt='%d')
    np.savetxt('../data/sim_test.txt', SIM, fmt='%d')
    write_leveldbs(X,Y, SIM, name='test')


if __name__ == "__main__":
    # save_pairs()
    test_cca()
