import numpy as np
import theano
import theano.tensor as T
from logistic_sgd import load_data
from dAE import dAE_nobias
import pickle



datasets = load_data('mnist.pkl.gz')
test_set_x, test_set_y = datasets[1]
print 'test shape: ',test_set_x.get_value(borrow=True).shape

with open('models/dae/dAE_mnist_full.pkl', 'rb') as input:
    da = pickle.load(input)

index = T.lscalar()
x = T.matrix('x')
def da_recon(da, x, corruption_level):
    tilde_x = da.get_corrupted_input(x, corruption_level)
    y = da.get_hidden_values(tilde_x)
    z = da.get_reconstructed_input(y)
    return z
#
# test_da = theano.function(
#     [index],
#     da_recon(da=da,x=x, corruption_level = .0),
#     givens={
#         x: test_set_x[index]
#     }
# )

z = da_recon(da,test_set_x[0], .3)
print 'Input: ', test_set_x[0]
print 'Recon: ',z
