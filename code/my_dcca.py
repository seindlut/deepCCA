
import os
import sys
import time
import gzip
import cPickle

import numpy as np

import theano
import theano.tensor as T
from numpy.linalg import svd
from scipy.linalg import sqrtm

from mlp import HiddenLayer, MLP

DEBUG = False

def mat_pow(matrix):
    return np.real(sqrtm(np.linalg.inv(matrix)))


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()


    def shared_dataset(data_xy, borrow=True):
        #import copy
        data_x, data_y = data_xy
        #daya_y = copy.deepcopy(data_x)

        # Binarizing the labels
        data_y_new = np.zeros((data_y.shape[0], data_y.max()+1))
        for i in range(data_y.shape[0]):
            data_y_new[i, data_y[i]] = 1
        data_y = data_y_new
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, shared_y
    train_set = (train_set[0][:100], train_set[1][:100])
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


class dCCA(object):
    def __init__(self,rng, net1, net2):
        self.net1 = net1
        self.net2 = net2

        self.cca = CCALayer(
            input1=self.net1.output,
            input2=self.net2.output,
        )

        self.correlation = (
            self.cca.correlation
        )

        self.params = self.net1.params + self.net2.params

    def get_cost_updates(self, H1,H1back, H2,H2back, learning_rate=0.1, momentum = 1):

        m = H1.shape[0]
        H1bar = H1 #- (1.0/m)*T.dot(H1, T.shared(np.ones((m,m))))
        H2bar = H2 #- (1.0/m)*T.dot(H1, T.ones_like(np.ones((m,m))))
        SigmaHat12 = (1.0/(m-1))*np.dot(H1bar, H2bar.T)
        SigmaHat11 = (1.0/(m-1))*np.dot(H1bar, H1bar.T)
        SigmaHat11 = SigmaHat11 + self.cca.r1*np.identity(SigmaHat11.shape[0])
        SigmaHat22 = (1.0/(m-1))*np.dot(H2bar, H2bar.T)
        SigmaHat22 = SigmaHat22 + self.cca.r2*np.identity(SigmaHat22.shape[0])
        # Square root of matrices:#
        # SigmaHat11_2 = SigmaHat11**(-0.5)
        # SigmaHat22_2 = SigmaHat22**(-0.5)
        SigmaHat11_2 = mat_pow(SigmaHat11)
        SigmaHat22_2 = mat_pow(SigmaHat22)


        Tval = np.dot(SigmaHat11_2, np.dot(SigmaHat12, SigmaHat22_2))
        corr = np.trace(np.dot(Tval.T, Tval))**(0.5)
        if DEBUG:
            print 'Sigma sqrt: '
            print '11: ', np.mean(SigmaHat11_2.flatten())
            print '22: ', np.mean(SigmaHat22_2.flatten())
            print '12: ', np.mean(SigmaHat12.flatten())

        U, D, V = svd(Tval,full_matrices=False)
        D = np.diag(D)
        V = V.T
        if DEBUG:
            print 'SVD:',
            print np.allclose(Tval, np.dot(U, np.dot(D, V.T))), np.linalg.norm(Tval - np.dot(U, np.dot(D, V.T)))

        UVT = np.dot(U, V.T)
        UDUT = np.dot(U, np.dot(D, U.T))

        Delta12 = np.dot(SigmaHat11_2, np.dot(UVT, SigmaHat22_2))
        Delta11 = -.5* np.dot(SigmaHat11_2, np.dot(UDUT, SigmaHat11_2))
        Delta22 = -.5* np.dot(SigmaHat22_2, np.dot(UDUT, SigmaHat22_2))

        # print 'Grad: '
        # print '11: ', Delta11.shape
        # print '22: ', Delta22.shape
        # print '12: ', Delta12.shape

        grad_Corr_to_inp1 = (1.0/(m-1)) * (2*np.dot(Delta11,H1bar)+np.dot(Delta12,H2bar))
        grad_Corr_to_inp2 = (1.0/(m-1)) * (2*np.dot(Delta22,H2bar)+np.dot(Delta12,H1bar))

        # print 'inp1: ', grad_Corr_to_inp1.shape
        # print 'inp2: ',grad_Corr_to_inp2.shape

        # The gradients wrt the CCAlayer parametres (W & b)
        gparam_W1 = (grad_Corr_to_inp1) * (H1*(1-H1))
        gparam_W1 = np.dot(H1back.T, gparam_W1)
        gparam_b1 = grad_Corr_to_inp1 * (H1*(1-H1))
        gparam_b1 = np.dot( gparam_b1.T, np.ones((m,1)))

        gparam_W2 = (grad_Corr_to_inp2) * (H2*(1-H2))
        gparam_W2 = np.dot(H2back.T, gparam_W2)
        gparam_b2 = grad_Corr_to_inp2 * (H2*(1-H2))
        gparam_b2 = np.dot( gparam_b2.T, np.ones((m,1)))

        if DEBUG:
            print 'gW1 :', np.mean(gparam_W1.flatten())
            print 'gB1 :', np.mean(gparam_b1.flatten())
            print 'gW2 :', np.mean(gparam_W2.flatten())
            print 'gB2 :', np.mean(gparam_b2.flatten())

        gparams1 = [theano.shared(np.array(gparam_W1,dtype=theano.config.floatX), borrow=True),
            theano.shared(np.array(gparam_b1.flatten(),dtype=theano.config.floatX), borrow=True) ]

        gparams2 = [theano.shared(np.array(gparam_W2,dtype=theano.config.floatX), borrow=True),
            theano.shared(np.array(gparam_b2.flatten(), dtype=theano.config.floatX), borrow=True) ]

        updates1 = [
            (param, momentum * param.eval() + learning_rate * gparam)
            for param, gparam in zip(self.net1.lastLayer.params, gparams1)
        ]

        updates2 = [
            (param, momentum * param.eval() + learning_rate * gparam)
            for param, gparam in zip(self.net2.lastLayer.params, gparams2)
        ]

        return updates1, updates2, corr


class CCALayer(HiddenLayer):
    def __init__(self, input1, input2):
        self.input1 = input1
        self.input2 = input2

        self.r1 = 0.001
        self.r2 = 0.001

    def correlation(self):
        H1 = self.input1
        H2 = self.input2
        m = H1.shape[1]
        H1bar = H1 #- (1.0/m)*T.dot(H1, T.shared(np.ones((m,m))))
        H2bar = H2 #- (1.0/m)*T.dot(H1, T.ones_like(np.ones((m,m))))
        SigmaHat12 = (1.0/(m-1))*T.dot(H1bar, H2bar.T)
        SigmaHat11 = (1.0/(m-1))*T.dot(H1bar, H1bar.T)
        SigmaHat11 = SigmaHat11 + self.r1*T.identity_like(SigmaHat11)
        SigmaHat22 = (1.0/(m-1))*T.dot(H2bar, H2bar.T)
        SigmaHat22 = SigmaHat22 + self.r2*T.identity_like(SigmaHat22)
        # Square root of matrices:
        SigmaHat11_2 = SigmaHat11**(-.5)
        SigmaHat22_2 = SigmaHat22**(-.5)

        Tval = T.dot(SigmaHat11_2, T.dot(SigmaHat12, SigmaHat22_2))
        corr = T.nlinalg.trace(T.dot(Tval.T, Tval))**(0.5)
        return -1*corr


def test_dcca(learning_rate= .2, momentum =1, L1_reg=1e-6, L2_reg=1e-6, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    # Loading datasets:
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x1 = T.matrix('x1')  # image
    x2 = T.matrix('x2')  # labels

    rng = np.random.RandomState(1234)

    # construct the MLP class
    net1 = MLP(
        rng = rng,
        input = x1,
        n_in = 28*28,
        n_hidden = 300,
        n_out = 8
    )

    net2 = MLP(
        rng= rng,
        input = x2,
        n_in = 10,
        n_hidden = 20,
        n_out = 8
    )

    N = dCCA(rng, net1, net2)

    # -------------------------------------------- Forward propagation funcions
    fprop1 = theano.function(
        inputs=[],
        outputs=(net1.output, net1.hiddenLayer.output),
        givens ={x1: train_set_x, x2: train_set_y}
    )
    fprop2 = theano.function(
        inputs=[],
        outputs= (net2.output, net2.hiddenLayer.output),
        givens = {x2: train_set_y, x1: train_set_x}
    )

    cost1 = (
        N.correlation() + L1_reg * net1.L1
        + L2_reg * net1.L2_sqr
    )

    cost2 = (
        N.correlation() + L1_reg * net2.L1
        + L2_reg * net2.L2_sqr
    )
    cost = N.correlation()

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'



    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        print 'epoch', epoch,
        H1, H1back = fprop1()
        H2 , H2back = fprop2()
        if DEBUG:
            print 'Out1: ',H1.shape
            print 'Back1: ',H1back.shape
            print 'Out2: ', H2.shape
            print 'Back2: ',H2back.shape
        # cc1 = theano.function(inputs= [], outputs = (cost1), givens={x1: train_set_x , x2: train_set_y})
        # print 'Net1 cost : ', cc1()
        # cc2 = theano.function(inputs= [], outputs = (cost2), givens={x1: train_set_x , x2: train_set_y})
        # print 'Net2 cost : ', cc2()
        # theano.printing.pydotprint(cc, outfile="models/dcca/correlation.png", var_with_name_simple=True)

        up = N.get_cost_updates(H1, H1back, H2 , H2back, learning_rate= learning_rate, momentum = momentum)
        # backprop = theano.function(inputs =[],
        #             outputs = up,
        #             givens ={x1: train_set_x, x2: train_set_y}
        #             )
        updates1, updates2, corr = up

        print 'Correlation :', corr
        gparams1 = [T.grad(cost1, param) for param in net1.hiddenLayer.params]
        backupdates1 = [
            (param, momentum * param - learning_rate * gparam)
            for param, gparam in zip(net1.hiddenLayer.params, gparams1)
        ]
        train_net1 = theano.function(
            inputs=[],
            outputs=[cost, cost1],
            updates=updates1 + backupdates1,
            givens={x1: train_set_x, x2: train_set_y}
        )
        gparams2 = [T.grad(cost2, param) for param in net2.hiddenLayer.params]
        backupdates2 = [
            (param, momentum * param - learning_rate * gparam)
            for param, gparam in zip(net2.hiddenLayer.params, gparams2)
        ]
        train_net2 = theano.function(
            inputs=[],
            outputs=[cost, cost2],
            updates=updates2 + backupdates2,
            givens={x1: train_set_x, x2: train_set_y}
        )

        cc, cs1 = train_net1()
        cc_, cs2 = train_net2()
        if epoch > 100:
            break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    test_dcca()
