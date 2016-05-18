"""
This file is part of deepcca.

deepcca is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

deepcca is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with deepcca.  If not, see <http://www.gnu.org/licenses/>.
"""
"""

Multi-layer Perceptron

References:

http://deeplearning.net/tutorial/mlp.html


"""

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


class DCCA(MLP):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.nnet.sigmoid,
        )

        self.lastLayer = CCALayer(
            rng=rng,
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            activation=T.nnet.sigmoid
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.lastLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.lastLayer.W ** 2).sum()
        )

        self.correlation = (
            self.lastLayer.correlation
        )
        self.correlation_numpy = (
            self.lastLayer.correlation_numpy
        )

        self.output = self.lastLayer.output
        self.params = self.hiddenLayer.params + self.lastLayer.params

    def get_updates(self, learning_rate=0.01):
        m = 10000
        U, V, D = theano.tensor.nlinalg.svd(self.lastLayer.Tval)
        UVT = T.dot(U, V.T)
        Delta12 = T.dot(self.lastLayer.SigmaHat11**(-0.5), T.dot(UVT, self.lastLayer.SigmaHat22**(-0.5)))
        UDUT = T.dot(U, T.dot(D, U.T))
        Delta11 = (-0.5) * T.dot(self.lastLayer.SigmaHat11**(-0.5), T.dot(UDUT, self.lastLayer.SigmaHat22**(-0.5)))
        grad_E_to_o = (1.0/(m-1)) * (2*Delta11*self.lastLayer.H1bar+Delta12*self.lastLayer.H2bar)

        # The gradients wrt the CCAlayer parametres (W & b)
        gparam_W = (grad_E_to_o) * (self.lastLayer.output*(1-self.lastLayer.output)) * (self.hiddenLayer.output)
        gparam_b = (grad_E_to_o) * (self.lastLayer.output*(1-self.lastLayer.output)) * theano.shared(numpy.array([1.0],dtype=theano.config.floatX), borrow=True)
        gparams = [gparam_W, gparam_b.flatten()]

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.lastLayer.params, gparams)
        ]

        return updates


class CCALayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.sigmoid):

        self.n_in = n_in
        self.n_out = n_out
        self.input = input
        self.activation = activation

        self.r1 = 0.001
        self.r2 = 0.001

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )

        self.params = [self.W, self.b]
        # self.params = [self.W]

    def correlation(self, H1, H2):
        #H1 = self.output.T
        m = H1.shape[1]
        H1bar = H1 #- (1.0/m)*T.dot(H1, T.shared(numpy.ones((m,m))))
        H2bar = H2 #- (1.0/m)*T.dot(H1, T.ones_like(numpy.ones((m,m))))
        SigmaHat12 = (1.0/(m-1))*T.dot(H1bar, H2bar.T)
        SigmaHat11 = (1.0/(m-1))*T.dot(H1bar, H1bar.T)
        SigmaHat11 = SigmaHat11 + self.r1*T.identity_like(SigmaHat11)
        SigmaHat22 = (1.0/(m-1))*T.dot(H2bar, H2bar.T)
        SigmaHat22 = SigmaHat22 + self.r2*T.identity_like(SigmaHat22)
        Tval = T.dot(SigmaHat11**(-0.5), T.dot(SigmaHat12, SigmaHat22**(-0.5)))
        corr = T.nlinalg.trace(T.dot(Tval.T, Tval))**(0.5)
        # Store the tensors for later use
        self.SigmaHat11 = SigmaHat11
        self.SigmaHat12 = SigmaHat12
        self.SigmaHat22 = SigmaHat22
        self.H1bar = H1bar
        self.H2bar = H2bar
        self.Tval = Tval
        return -1*corr

    def correlation_numpy(self, H1, H2):
        m = H1.shape[1]

        H1bar = H1#.eval() #- (1.0/m)*numpy.dot(H1, numpy.ones((m,m), dtype=numpy.float32))
        H2bar = H2#.eval() #- (1.0/m)*numpy.dot(H2, numpy.ones((m,m), dtype=numpy.float32))
        SigmaHat12 = (1.0/(m-1))*numpy.dot(H1bar, H2bar.T)
        SigmaHat11 = (1.0/(m-1))*numpy.dot(H1bar, H1bar.T)
        SigmaHat11 = SigmaHat11 + 0.0001*numpy.identity(SigmaHat11.shape[0], dtype=numpy.float32)
        SigmaHat22 = (1.0/(m-1))*numpy.dot(H2bar, H2bar.T)
        SigmaHat22 = SigmaHat22 + 0.0001*numpy.identity(SigmaHat22.shape[0], dtype=numpy.float32)

        Tval = numpy.dot(mat_pow(SigmaHat11), numpy.dot(SigmaHat12, mat_pow(SigmaHat22)))

        corr =  numpy.trace(numpy.dot(Tval.T, Tval))**(0.5)
        return corr, Tval


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
    x1 = T.matrix('x1')  # the data is presented as rasterized images
    x2 = T.matrix('x2')  # the labels are presented as 1D vector of
                        # [int] labels
    h1 = T.matrix('h1')  # the data is presented as rasterized images
    h2 = T.matrix('h2')  # the labels are presented as 1D vector of

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    net1 = DCCA(
        rng=rng,
        input=x1,
        n_in=28 * 28,
        n_hidden=300,
        n_out=8,
    )
    net2 = DCCA(
        rng=rng,
        input=x2,
        n_in=10,
        n_hidden=20,
        n_out=8,
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
