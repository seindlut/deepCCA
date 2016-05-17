import numpy as np
import theano
import theano.tensor as T
from logistic_sgd import load_data
from dAE import dAE_nobias, dAE
from six.moves import cPickle




datasets = load_data('mnist.pkl.gz')
test_set_x, test_set_y = datasets[2]

""" DAE """
with open('models/dae/dAE_mnist_corrupted_full.pkl', 'rb') as input:
    da = cPickle.load(input)

index = T.lscalar()
x = T.matrix('x')

def da_recon(input, corruption_level):
    tilde_x = da.get_corrupted_input(input, corruption_level)
    y = da.get_hidden_values(tilde_x)
    z = da.get_reconstructed_input(y)
    return z

test_da = theano.function(
    [index],
    da_recon(x, corruption_level = .0),
    givens={
        x: test_set_x[index:index+1]
    }
)

z = test_da(0)
# for plotting
with open('models/dae/dAE_test_0.pkl', 'wb') as output:
    cPickle.dump(z, output, cPickle.HIGHEST_PROTOCOL)
