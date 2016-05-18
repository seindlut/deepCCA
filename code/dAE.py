"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""
from __future__ import division
from utils import *
import os
import sys
import time
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from logistic_sgd import load_data

from six.moves import cPickle
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


def Trelu(x): #-----------------------------------------Activation function
    return theano.tensor.switch(x<1e-06, 1e-06, x)
    #return T.maximum(0,x)
    #return x * (x > 0)
    #return T.nnet.sigmoid(x)
    #return T.tanh(x)


class dAE(object):
    """Denoising Auto-Encoder class (dAE)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        z = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:# ------------------------------------------------------------------------------------ Random initialization
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy.random.normal(loc=0., scale=.01, size=(n_visible, n_hidden)),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                )+numpy.random.normal(loc=0., scale=.001, size=(n_visible)),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                )+numpy.random.normal(loc=0., scale=.001, size=(n_hidden)),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        self.L1 = (
            abs(self.W).sum()
        )
        self.L2_sqr = (
            ((self.W**2)).sum()
        )
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level): # ------------------------------------------  Input corruption
        """
        This function keeps (1-corruption_level) entries of the inputs and zero-out randomly selected subset
        of size coruption_level
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - corruption_level and 0 with
                corruption_level

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        # return Trelu(T.dot(input, self.W) + self.b)
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
        # return Trelu(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        #---------------------------------------------------- ------ Cross entropy loss
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # --------------------------------------------------------- Eucliden loss
        # L= T.mean((z-self.x)**2)
        # note : L is now a vector, where each element is the
        #        cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch

        # cost = T.mean(L)
        L1_reg = 0.001
        L2_reg = 0.001
        cost = T.mean(L) + L1_reg * self.L1 + L2_reg * self.L2_sqr

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam) # ------------------------------------------------------------ Simple SGD
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

    def mse_test_recon(self, corruption_level):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        return T.mean((z-self.x)**2)



""" Same dAE without bias terms """
class dAE_nobias(object):
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy.random.normal(loc=0., scale=.01, size=(n_visible, n_hidden)),

                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        self.W = W
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        self.L1 = (
            # abs(self.W*(self.W<0)).sum()
            abs(self.W).sum()
        )

        self.L2_sqr = (
            # ((self.W**2)*(self.W<0)).sum()
            ((self.W**2)).sum()
        )
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        y = self.get_hidden_values(self.x)
        self.output = self.get_reconstructed_input(y)
        self.params = [self.W]

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        # return Trelu(T.dot(input, self.W))
        return T.nnet.sigmoid(T.dot(input, self.W))

    def get_reconstructed_input(self, hidden):
        # return Trelu(T.dot(hidden, self.W_prime))
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime))

    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        self.z = z
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # L=T.mean((z-self.x)**2)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch

        L1_reg = 0.0001
        L2_reg = 0.000
        cost = T.mean(L) + L1_reg * self.L1 + L2_reg * self.L2_sqr

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

    def mse(self):
        return T.mean((self.z-self.x)**2)

    def mse_test_recon(self, corruption_level):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        return T.mean((z-self.x)**2)



def test_dAE(learning_rate=0.1, training_epochs=100, dataset='full', batch_size=64, output_folder='models/dae'):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    if dataset =='halves':
        datasets = load_data_half('mnist.pkl.gz')
        dim = (28,28/2)
    elif dataset=='pairs':
        datasets = load_data_full()
        dim = (28,28)
    elif dataset =='full' :
        datasets = load_data('mnist.pkl.gz')
        dim = (28,28)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches =  test_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    #-------------------------------------------------
    # BUILDING THE MODEL NO CORRUPTION & NO BIAS
    #-------------------------------------------------

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dAE_nobias(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible= dim[0]*dim[1],
        n_hidden=30
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        da.mse(),
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_da = theano.function(
        [index],
        da.mse_test_recon(0.),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )


    start_time = time.clock()

    #--------------
    # TRAINING
    #--------------
    # go through training epochs
    mse_train =[]
    mse_test = []
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index)/batch_size)

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
        mse_train.append(numpy.mean(c))
        # ---------------------------------------------------------- Test
        cc =[]
        for ii in xrange(n_test_batches):
            cc.append(test_da(ii)/batch_size)
        print 'Training epoch %d, test cost ' % epoch, numpy.mean(cc)
        mse_test.append(numpy.mean(cc))

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))

    plt.figure(figsize=(6,4))
    plt.plot(range(len(mse_train)), mse_train, label ='Train')
    plt.plot(range(len(mse_test)), mse_test, label ='Test')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(output_folder+'/f30_unc_log_train.pdf',bbox_inches='tight')

    image = Image.fromarray(
        tile_raster_images(X=da.W.get_value(borrow=True).T,
                           img_shape=dim, tile_shape=(6, 5),
                           tile_spacing=(1, 1)))
    image.save(output_folder+'/f30_filters_corruption_0.png')

    with open(output_folder+'/f30_unc_'+dataset+'.pkl', 'wb') as output:
        cPickle.dump(da, output, cPickle.HIGHEST_PROTOCOL)



    #-------------------------------------------------
    # BUILDING THE CORRUPTED MODEL W/ BIAS
    #-------------------------------------------------

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dAE(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=dim[0]*dim[1],
        n_hidden=30
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_da = theano.function(
        [index],
        da.mse_test_recon(0.3),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = time.clock()

    ############
    # TRAINING #
    ############
    mse_train =[]
    mse_test =[]
    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
        mse_train.append(numpy.mean(c))
        mse_test.append(numpy.mean(test_da()))
    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The 30% corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % (training_time / 60.))

    plt.figure(figsize=(6,4))
    plt.plot(range(len(mse_train)), mse_train, label ='Train')
    plt.plot(range(len(mse_test)), mse_test, label ='Test')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(output_folder+'/f30_corr30_log_train.pdf',bbox_inches='tight')


    image = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=dim, tile_shape=(6,5),
        tile_spacing=(1, 1)))
    image.save(output_folder+'/filters_corr30.png')

    # Save the model for later use:
    with open(output_folder+'/f30_corr30_'+dataset+'.pkl', 'wb') as output:
        cPickle.dump(da, output, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    test_dAE()
