"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

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
import sys
import time
import os
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mlp import HiddenLayer
from dA import dA, dA_nobias
from utils import tile_raster_images,plot_weights, load_data_half

try:
    from matplotlib import pyplot as pp
    import matplotlib.cm as cm

except ImportError:
    print 'matplotlib is could not be imported'

def Trelu(x):
    #return theano.tensor.switch(x<1e-06, 1e-06, x)
    return T.nnet.sigmoid(x)
    #return T.tanh(x)

# start-snippet-1
class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=28*28//2,
        hidden_layers_sizes=[500, 500],
        corruption_levels=[0.1, 0.1]
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        ##activation=T.nnet.sigmoid)
                                        activation=Trelu)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            #dA_layer = dA(numpy_rng=numpy_rng,
                          #theano_rng=theano_rng,
                          #input=layer_input,
                          #n_visible=input_size,
                          #n_hidden=hidden_layers_sizes[i],
                          #W=sigmoid_layer.W,
                          #bhid=sigmoid_layer.b)
            dA_layer = dA_nobias(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W
                          )
            self.dA_layers.append(dA_layer)
        # end-snippet-2
        
        #self.errors = self.logLayer.errors(self.x)

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

   
class SdA_regress(object):
    def __init__(
        self,
        SdA_inp,
        SdA_out,
        log_reg,
        numpy_rng,
        theano_rng=None,
        n_inp=28*28//2,
        n_out=28*28//2,
        hidden_layers_sizes_inp=[500, 500],
        hidden_layers_sizes_out=[500, 500],
        corruption_levels_inp=[0.1, 0.1],
        corruption_levels_out=[0.1, 0.1]
    ):
        """ This class is made to support a variable number of layers.
   
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights
   
        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`
   
        :type n_ins: int
        :param n_ins: dimension of the input to the sdA
   
        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value
   
        :type n_outs: int
        :param n_outs: dimension of the output of the network
   
        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """
        self.SdA_inp = SdA_inp
        self.SdA_out = SdA_out
        self.log_reg = log_reg
        self.sigmoid_layers = []
        self.params = []
        hidden_layers_sizes = hidden_layers_sizes_inp + hidden_layers_sizes_out[::-1]
        
        # +1 is for logistic regression layer
        self.n_layers = len(hidden_layers_sizes)
   
        assert self.n_layers > 0
   
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.matrix('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        
        # end-snippet-1
   
        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP
   
        # start-snippet-2
        for i in xrange(self.n_layers+1):
            # construct the sigmoidal layer
   
            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_inp
            else:
                input_size = hidden_layers_sizes[i - 1]
            
            if i == self.n_layers:
                output_size = n_out
            else:
                output_size = hidden_layers_sizes[i]
                
            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
            #elif i < len(hidden_layers_sizes_inp):
            #    layer_input = self.sigmoid_layers[-1].output
            #elif i == len(hidden_layers_sizes_inp):
            #    layer_input = self.logistic_layer.output
                
            if i < len(hidden_layers_sizes_inp):
                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                            input=layer_input,
                                            n_in=input_size,
                                            n_out=output_size,
                                            activation=T.nnet.sigmoid,
                                            W=theano.shared(self.SdA_inp.dA_layers[i].W.eval()),
                                            b=theano.shared(self.SdA_inp.dA_layers[i].b.eval()))
            elif i == len(hidden_layers_sizes_inp): # logistic_reg layer
                #self.log_reg = LogisticRegression(layer_input,
                                            #input_size,
                                            #hidden_layers_sizes[i])
                #self.log_reg = HiddenLayer(rng=numpy_rng,
                                            #input=self.h1,
                                            #n_in=input_size,
                                            #n_out=output_size,
                                            #activation=T.nnet.sigmoid                                                                                       
                                            #)
                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                            input=layer_input,
                                            n_in=input_size,
                                            n_out=output_size,
                                            activation=T.nnet.sigmoid,
                                            W=theano.shared(log_reg.W.eval()),
                                            b=theano.shared(log_reg.b.eval())
                                            )
            else:
                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                            input=layer_input,
                                            n_in=input_size,
                                            n_out=output_size,
                                            activation=T.nnet.sigmoid,
                                            W=theano.shared(self.SdA_out.dA_layers[-1*(i-len(hidden_layers_sizes_inp))].W.T.eval()),
                                            b=theano.shared(self.SdA_out.dA_layers[-1*(i-len(hidden_layers_sizes_inp))].b_prime.eval()))
                
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)
   
            # Construct a denoising autoencoder that shared weights with this
            # layer            
      
        # end-snippet-2
        self.finetune_cost = self.sigmoid_layers[-1].mse(self.y)
        self.errors = self.sigmoid_layers[-1].mse(self.y)
   
    def build_middle_pretrain(self, datasets, batch_size, learning_rate):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        if 1: # for middle layer
            
            fprop_inp = theano.function(
                [],
                self.SdA_inp.sigmoid_layers[-1].output,
                givens={
                    self.SdA_inp.sigmoid_layers[0].input: test_set_x
                },
                name='fprop_inp'
            )
            fprop_out = theano.function(
                [],
                self.SdA_out.sigmoid_layers[-1].output,
                givens={
                    self.SdA_out.sigmoid_layers[0].input: test_set_y
                },
                name='fprop_out'
            )
            H1=fprop_out() 
            H2=fprop_out()
            H1=theano.shared(H1)
            H2=theano.shared(H2)
            # compute the gradients with respect to the model parameters
            self.logreg_cost = self.log_reg.mse(self.h2)

            gparams = T.grad(self.logreg_cost, self.log_reg.params)
    
            # compute list of fine-tuning updates
            updates = [
                (param, param - gparam * learning_rate)
                for param, gparam in zip(self.params, gparams)
            ]

            train_fn_middle = theano.function(
                inputs=[index],
                outputs=self.logreg_cost,
                updates=updates,
                givens={
                    self.h1: H1[
                        index * batch_size: (index + 1) * batch_size
                    ],
                    self.h2: H2[
                        index * batch_size: (index + 1) * batch_size
                    ]
                },
                name='train_middle'
            )
        
        return train_fn_middle
        
    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]
        train_set_x=train_set_x.eval()
        train_set_y=train_set_y.eval()
        train_set_x_lab=train_set_x[:1000,:]
        train_set_x_unlab=train_set_x[1000:,:]
        train_set_y_lab=train_set_y[:1000,:]
        train_set_y_unlab=train_set_y[1000:,:]
        train_set_x_lab=theano.shared(numpy.asarray(train_set_x_lab,
                                                    dtype=theano.config.floatX),
                                      borrow=True)
        train_set_y_lab=theano.shared(numpy.asarray(train_set_y_lab,
                                                    dtype=theano.config.floatX),
                                      borrow=True)
        train_set_x_unlab=theano.shared(numpy.asarray(train_set_x_unlab,
                                                      dtype=theano.config.floatX),
                                        borrow=True)
        train_set_y_unlab=theano.shared(numpy.asarray(train_set_y_unlab,
                                                      dtype=theano.config.floatX),
                                        borrow=True)
        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch
        if 0: # for middle layer
            
            fprop_inp = theano.function(
                [],
                self.SdA_inp.sigmoid_layers[-1].output,
                givens={
                    self.SdA_inp.sigmoid_layers[0].input: test_set_x
                },
                name='fprop_inp'
            )
            fprop_out = theano.function(
                [],
                self.SdA_out.sigmoid_layers[-1].output,
                givens={
                    self.SdA_out.sigmoid_layers[0].input: test_set_y
                },
                name='fprop_out'
            )
            H1=fprop_out() 
            H2=fprop_out()
            H1=theano.shared(H1)
            H2=theano.shared(H2)
            # compute the gradients with respect to the model parameters
            self.logreg_cost = self.log_reg.mse(self.h2)

            gparams = T.grad(self.logreg_cost, self.log_reg.params)
    
            # compute list of fine-tuning updates
            updates = [
                (param, param - gparam * learning_rate)
                for param, gparam in zip(self.params, gparams)
            ]

            train_fn_middle = theano.function(
                inputs=[],
                outputs=self.logreg_cost,
                updates=updates,
                givens={
                    self.h1: H1,
                    self.h2: H2
                },
                name='train_middle'
            )
        
        
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]
        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x_lab[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y_lab[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [],
            self.sigmoid_layers[-1].mse(self.y),
            givens={
                self.x: test_set_x,
                self.y: test_set_y
            },
            name='test'
        )

        valid_score_i = theano.function(
            [],
            self.sigmoid_layers[-1].mse(self.y),
            givens={
                self.x: valid_set_x,
                self.y: valid_set_y
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score_i, test_score_i


    
def test_SdA_regress(finetune_lr=0.05, pretraining_epochs=10,
             pretrain_lr=0.1, training_epochs=10000,
             dataset='mnist.pkl.gz', batch_size=20):
    datasets = load_data_half(dataset)

    train_set_x, train_set_y = datasets[0]##
    valid_set_x, valid_set_y = datasets[1]##
    test_set_x, test_set_y = datasets[2]##
    train_set_x=train_set_x.eval()
    train_set_y=train_set_y.eval()
    import theano
    train_set_x_lab=train_set_x[:,:]
    train_set_x_unlab=train_set_x[:,:]
    train_set_y_lab=train_set_y[:,:]
    train_set_y_unlab=train_set_y[:,:]
    train_set_x_lab=theano.shared(numpy.asarray(train_set_x_lab,
                                                dtype=theano.config.floatX),
                                  borrow=True)
    train_set_y_lab=theano.shared(numpy.asarray(train_set_y_lab,
                                                dtype=theano.config.floatX),
                                  borrow=True)
    train_set_x_unlab=theano.shared(numpy.asarray(train_set_x_unlab,
                                                  dtype=theano.config.floatX),
                                    borrow=True)
    train_set_y_unlab=theano.shared(numpy.asarray(train_set_y_unlab,
                                                  dtype=theano.config.floatX),
                                    borrow=True)

    # compute number of minibatches for training, validation and testing
    n_train_batches_l = train_set_y_lab.eval().shape[0]
    n_train_batches_l /= batch_size
    n_train_batches_u = train_set_y_unlab.eval().shape[0]
    n_train_batches_u /= batch_size
    # compute number of minibatches for training, validation and testing
    #n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    #n_train_batches /= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    #from SdA_orig import SdA as SdA_old
    hidden_layer_size = 100
    SdA_inp = SdA(numpy_rng,
                  n_ins=392,
                  hidden_layers_sizes=[hidden_layer_size]
    )
    SdA_out = SdA(numpy_rng,
                  n_ins=392,
                  hidden_layers_sizes=[hidden_layer_size]
    )
        
    # PRETRAINING THE MODEL #
    if 0 : # pretrain inp ae
        print '... getting the pretraining functions for INPUT AE'
        pretraining_fns = SdA_inp.pretraining_functions(train_set_x=train_set_x_unlab,
                                                    batch_size=batch_size)
    
        print '... pre-training the model'
        start_time = time.clock()
        ## Pre-train layer-wise
        corruption_levels = [.1, .2, .3]
        for i in xrange(SdA_inp.n_layers):
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches_u):
                    c.append(pretraining_fns[i](index=batch_index,
                             corruption=corruption_levels[i],
                             lr=pretrain_lr))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)
    
        end_time = time.clock()
    
        print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
    if 0 : # pretrain out ae
        print '... getting the pretraining functions for OUTPUT AE'
        pretraining_fns = SdA_out.pretraining_functions(train_set_x=train_set_y_unlab,
                                                    batch_size=batch_size)
    
        print '... pre-training the model'
        start_time = time.clock()
        ## Pre-train layer-wise
        corruption_levels = [.5, .2, .3]
        for i in xrange(SdA_out.n_layers):
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches_u):
                    c.append(pretraining_fns[i](index=batch_index,
                             corruption=corruption_levels[i],
                             lr=pretrain_lr))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)
    
        end_time = time.clock()
    
        print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
        
    if 0: # save aes
        f=open('aes_shallow_sig_nobias.pkl', 'w+')
        import pickle
        pickle.dump(SdA_inp, f)
        pickle.dump(SdA_out, f)
        f.flush()
        f.close() 
    if 0: # load aes
        f=open('aes_shallow_sig_nobias.pkl', 'r')
        import pickle
        SdA_inp=pickle.load(f)
        SdA_out=pickle.load(f)
        f.close()    
   
    if 1: # cca
        from dcca_numpy import netCCA_nobias, netCCA, dCCA
        from mlp_numpy import expit, logistic_prime, linear, linear_prime, relu, relu_prime, tanh, tanh_prime
        train_y1 = train_set_x_lab.eval()
        train_y2 = train_set_y_lab.eval()
        test_y1 = test_set_x.eval()
        test_y2 = test_set_y.eval()

        ##param1=((train_y1.shape[1],0,0),(2038, relu, relu_prime),(50, relu, relu_prime))
        ##param2=((train_y2.shape[1],0,0),(1608, relu, relu_prime),(50, relu, relu_prime))
        param1=((train_y1.shape[1],0,0),(hidden_layer_size, expit, logistic_prime))
        param2=((train_y2.shape[1],0,0),(hidden_layer_size, expit, logistic_prime))
        W1s = []
        b1s = []
        for i in range(len(SdA_inp.dA_layers)):
            W1s.append( SdA_inp.dA_layers[i].W.T.eval() )
            ##b1s.append( SdA_inp.dA_layers[i].b.eval() )
            ##b1s[-1] = b1s[-1].reshape((b1s[-1].shape[0], 1))
        W2s = []
        b2s = []
        for i in range(len(SdA_out.dA_layers)):
            W2s.append( SdA_out.dA_layers[i].W.T.eval() )
            ##b2s.append( SdA_out.dA_layers[i].b.eval() )
            ##b2s[-1] = b2s[-1].reshape((b2s[-1].shape[0], 1))

        numpy.random.seed(0)
        N1=netCCA_nobias(train_y1,param1, W1s)
        N2=netCCA_nobias(train_y2,param2, W2s)
        N = dCCA(train_y1, train_y2, N1, N2)
        N1.reconstruct(test_set_x.eval()[0,:])
        cnt = 0
        from dcca_numpy import cca_cost, cca, order_cost, cor_cost
        while True:
            X=N1.predict(test_set_x.eval())
            Y=N2.predict(test_set_y.eval())
            _H1 = numpy.dot(X, N.A1)
            _H2 = numpy.dot(Y, N.A2)
            print '****', cnt, cor_cost(_H1, _H2)
            X1_rec = numpy.tanh(X.dot(N1.weights[0]))
            X2_rec = numpy.tanh(Y.dot(N2.weights[0]))
            param=((hidden_layer_size,0,0),(hidden_layer_size, relu, relu_prime))
            from mlp_numpy import NeuralNetwork as NN

            lr=NN(X,Y,param)
            lr.train(X[:,:],Y[:,:],10, 0.005)
            Yh=lr.predict(X[:,:])
            X2_reg = N2.fs[-1](numpy.dot(Yh,N2.weights[0]))

            #X2_reg = N2.fs[-1](numpy.dot(_H1.dot(numpy.linalg.inv(N.A1)),N2.weights[0]))

            print '****', 'mse1:', numpy.mean((X1_rec-test_set_x.eval())**2.0)
            print '****', 'mse2:', numpy.mean((X2_rec-test_set_y.eval())**2.0)
            print '****', 'mse_map:', numpy.mean((X2_reg-test_set_y.eval())**2.0)

            if cnt % 2:
                N.train(5, True, 10000.0)
            else:
                N.train(5, False, 10000.0)

            cnt += 1
            f=open('netcca.pkl', 'w+')
            import pickle
            pickle.dump(N, f)
            pickle.dump(N, f)
            f.flush()
            f.close() 
            if cnt == 200:
                break
        for i in range(len(SdA_inp.dA_layers)):
            SdA_inp.dA_layers[i].W = theano.shared( N1.weights[i].T )
            SdA_inp.dA_layers[i].b = theano.shared( N1.biases[i][:,0] )
        
        for i in range(len(SdA_out.dA_layers)):
            SdA_out.dA_layers[i].W = theano.shared( N2.weights[i].T )
            SdA_out.dA_layers[i].b = theano.shared( N2.weights[i][:,0] )

        
    if 1 : # pretrain middle layer
        print '... pre-training MIDDLE layer'

        h1 = T.matrix('x')  # the data is presented as rasterized images
        h2 = T.matrix('y')  # the labels are presented as 1D vector of
        log_reg = HiddenLayer(numpy_rng, h1, hidden_layer_size, hidden_layer_size)

        if 1: # for middle layer
            learning_rate = 0.01
            fprop_inp = theano.function(
                [],
                SdA_inp.sigmoid_layers[-1].output,
                givens={
                    SdA_inp.sigmoid_layers[0].input: train_set_x_lab
                },
                name='fprop_inp'
            )
            fprop_out = theano.function(
                [],
                SdA_out.sigmoid_layers[-1].output,
                givens={
                    SdA_out.sigmoid_layers[0].input: train_set_y_lab
                },
                name='fprop_out'
            )
            #H11=fprop_inp() 
            #H21=fprop_out()
            ##H1=N1.predict(train_set_x.eval())
            ##H2=N2.predict(train_set_y.eval())
            H1=fprop_inp()
            H2=fprop_out()
            H1=theano.shared(H1)
            H2=theano.shared(H2)
            # compute the gradients with respect to the model parameters
            logreg_cost = log_reg.mse(h2)

            gparams = T.grad(logreg_cost, log_reg.params)
    
            # compute list of fine-tuning updates
            updates = [
                (param, param - gparam * learning_rate)
                for param, gparam in zip(log_reg.params, gparams)
            ]

            train_fn_middle = theano.function(
                inputs=[],
                outputs=logreg_cost,
                updates=updates,
                givens={
                    h1: H1,
                    h2: H2
                },
                name='train_middle'
            )
        epoch = 0
        while epoch < 10:
            print epoch, train_fn_middle()
            epoch += 1
            
    sda = SdA_regress(
        SdA_inp,
        SdA_out,
        log_reg,
        numpy_rng=numpy_rng,
        n_inp=28*28//2,
        hidden_layers_sizes_inp=[hidden_layer_size],
        hidden_layers_sizes_out=[hidden_layer_size],
        n_out=28*28//2
    )
    # end-snippet-3 start-snippet-4
    # end-snippet-4
    
    # FINETUNING THE MODEL #

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )
    
        
    print '... finetunning the model'
    # early-stopping parameters
    patience = 10 * n_train_batches_l  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches_l, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    fprop = theano.function(
        [],
        sda.sigmoid_layers[-1].output,
        givens={
            sda.x: test_set_x
        },
        name='fprop'
    )
    while True:
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches_l):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches_l + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches_l,
                       this_validation_loss ))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches_l,
                           test_score ))

            if patience <= iter:
                done_looping = True
                #break
            if 0: # vis weights
                fprop = theano.function(
                    [],
                    sda.sigmoid_layers[-1].output,
                    givens={
                        sda.x: test_set_x
                    },
                    name='fprop'
                )
                yh=fprop()
                yh=yh
    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss , best_iter + 1, test_score)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_SdA_regress()