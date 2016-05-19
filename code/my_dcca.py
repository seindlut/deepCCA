
import os
import sys
import time
import gzip
import cPickle

import numpy

import theano
import theano.tensor as T

import scipy.linalg

from mlp import load_data, HiddenLayer, MLP
def mat_pow(matrix):
    return scipy.linalg.sqrtm(numpy.linalg.inv(matrix))


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

    def get_updates(self, learning_rate=0.01):
        m = 10000
        U, V, D = theano.tensor.nlinalg.svd(self.cca.Tval)
        UVT = T.dot(U, V.T)
        UDUT = T.dot(U, T.dot(D, U.T))

        Delta12 = T.dot(self.cca.SigmaHat11_2, T.dot(UVT, self.cca.SigmaHat22_2))
        Delta11 = -.5* T.dot(self.cca.SigmaHat11_2, T.dot(UDUT, self.cca.SigmaHat11_2))
        Delta22 = -.5* T.dot(self.cca.SigmaHat22_2, T.dot(UDUT, self.cca.SigmaHat22_2))
        grad_E_to_o = (1.0/(m-1)) * (2*Delta11*self.cca.H1bar+Delta12*self.cca.H2bar)

        # The gradients wrt the CCAlayer parametres (W & b)
        gparam_W = (grad_E_to_o) * (self.cca.output*(1-self.cca.output)) * (self.hiddenLayer.output)
        gparam_b = (grad_E_to_o) * (self.cca.output*(1-self.cca.output)) * theano.shared(numpy.array([1.0],dtype=theano.config.floatX), borrow=True)
        gparams = [gparam_W, gparam_b.flatten()]

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.cca.params, gparams)
        ]

        return updates


class CCALayer(HiddenLayer):
    def __init__(self, input1, input2):
        self.input1 = input1
        self.input2 = input2

        self.r1 = 0.001
        self.r2 = 0.001

    def correlation(self):
        H1 = self.input1.T
        H2 = self.input2.T
        m = H1.shape[1]
        H1bar = H1 #- (1.0/m)*T.dot(H1, T.shared(numpy.ones((m,m))))
        H2bar = H2 #- (1.0/m)*T.dot(H1, T.ones_like(numpy.ones((m,m))))
        SigmaHat12 = (1.0/(m-1))*T.dot(H1bar, H2bar.T)
        SigmaHat11 = (1.0/(m-1))*T.dot(H1bar, H1bar.T)
        SigmaHat11 = SigmaHat11 + self.r1*T.identity_like(SigmaHat11)
        SigmaHat22 = (1.0/(m-1))*T.dot(H2bar, H2bar.T)
        SigmaHat22 = SigmaHat22 + self.r2*T.identity_like(SigmaHat22)
        # Square root of matrices:#FIXME
        SigmaHat11_2 = SigmaHat11**(-0.5)
        SigmaHat22_2 = SigmaHat22**(-0.5)

        Tval = T.dot(SigmaHat11_2, T.dot(SigmaHat12, SigmaHat22_2))
        corr = T.nlinalg.trace(T.dot(Tval.T, Tval))**(0.5)
        # Store the tensors for later use
        self.SigmaHat11_2 = SigmaHat11_2
        self.SigmaHat12_2 = SigmaHat12_2
        self.SigmaHat22 = SigmaHat22
        self.H1bar = H1bar
        self.H2bar = H2bar
        self.Tval = Tval
        return -1*corr


def test_dcca(learning_rate=0.01, L1_reg=0.0001, L2_reg=0.0001, n_epochs=1000,
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

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    net1 = MLP(
        rng = rng,
        input = x1,
        n_in = 28*28,
        n_hidden = 300,
        n_out = 8
    )

    net2 = MLP(
        rng = rng,
        input = x1,
        n_in = 10,
        n_hidden = 20,
        n_out = 8
    )


    # -------------------------------------------- Forward propagation funcions
    fprop_model1 = theano.function(
        inputs=[],
        outputs=(net1.hiddenLayer.output, net1.output),
        givens={
            x1: train_set_x
        }
    )
    fprop_model2 = theano.function(
        inputs=[],
        outputs=(net2.hiddenLayer.output, net2.output),
        givens={
            x2: train_set_y
        }
    )
    cost1 = (
        net1.correlation(h1, h2)
        + L1_reg * net1.L1
        + L2_reg * net1.L2_sqr
    )

    cost2 = (
        net2.correlation(h1, h2)
        + L1_reg * net2.L1
        + L2_reg * net2.L2_sqr
    )


    theano.printing.pydotprint(fprop_model2, outfile="models/dcca/model.png", var_with_name_simple=True)
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

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        print 'epoch', epoch
        print '...... Forward'
        h1hidden, h1cca = fprop_model1() # hidden and last layers outputs
        h2hidden, h2cca = fprop_model2()
        print h1cca
        print '...... Backward'
        #compute cost(H1, H2)
        H1 = h1cca.T
        H2 = h2cca.T
        print 'H1: ', H1.shape
        print 'H2: ', H2.shape
        corr1 = net1.correlation(H1,H2)
        print 'Net1 correlation : ', corr1.eval()
        corr2 = net2.correlation(H1,H2)
        print 'Net2 correlation: ',corr2.eval()
        assert(corr1.eval()==corr2.eval())


        h1tmp = theano.shared(numpy.asarray(net1.lastLayer.H1bar,dtype=theano.config.floatX),
                                 borrow=True)
        h2tmp = theano.shared(numpy.asarray(net2.lastLayer.H2bar,dtype=theano.config.floatX),
                                 borrow=True)

        train_model1 = theano.function(
            inputs=[],
            outputs=[],
            updates=net1.get_updates(learning_rate=0.01),
            givens={x1: train_set_x, h1:h1tmp, h2: h2tmp}
        )
        print '\n\n h1:', h1.eval()

        train_model2 = theano.function(
            inputs=[],
            outputs=[],
            updates=net2.get_updates(learning_rate=0.01),
            givens={x2: train_set_y, h2: h2tmp, h1: h1tmp }
        )

        minibatch_avg_cost1 = train_model1()
        minibatch_avg_cost2 = train_model2()
        print 'corr1', minibatch_avg_cost1
        print 'corr2', minibatch_avg_cost2
        print 'corr', corr
        if epoch > 10:
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
