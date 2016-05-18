
from scipy.special import expit
import numpy as np
from logistic_sgd import load_data
import sklearn.cross_decomposition
from cca_linear import cca as CCA
from dcca_utils import *
from mlp_numpy import *
from utils import load_data_half

class netCCA(object):
    def __init__(self, X, parameters,Ws=None, bs=None):
        #Input data
        self.X=X

        #Expect parameters to be a tuple of the form:
        #    ((n_input,0,0), (n_hidden_layer_1, f_1, f_1'), ...,
        #     (n_hidden_layer_k, f_k, f_k'), (n_output, f_o, f_o'))
        self.n_layers = len(parameters)
        #Counts number of neurons without bias neurons in each layer.
        self.sizes = [layer[0] for layer in parameters]
        #Activation functions for each layer.
        self.fs =[layer[1] for layer in parameters]
        #Derivatives of activation functions for each layer.
        self.fprimes = [layer[2] for layer in parameters]
        if Ws is None or bs is None:
            self.build_network()
        else:
            self.import_weights(Ws, bs)

    def import_weights(self, Ws, bs):
        #List of weight matrices taking the output of one layer to the input of the next.
        self.weights=[]
        #Bias vector for each layer.
        self.biases=[]
        self.weights_batch=[]
        #Bias vector for each layer.
        self.biases_batch=[]
        #Input vector for each layer.
        self.inputs=[]
        #Output vector for each layer.
        self.outputs=[]
        #Vector of errors at each layer.
        self.errors=[]
        #We initialise the weights randomly, and fill the other vectors with 1s.
        for layer in range(self.n_layers-1):
            n = self.sizes[layer]
            m = self.sizes[layer+1]
            self.weights.append(Ws[layer])
            self.biases.append(bs[layer])
            self.weights_batch.append(np.random.normal(0,1, (m,n)))
            self.biases_batch.append(np.random.normal(0,1,(m,1)))
            self.inputs.append(np.zeros((n,1),dtype=np.float32))
            self.outputs.append(np.zeros((n,1),dtype=np.float32))
            self.errors.append(np.zeros((n,1),dtype=np.float32))
        #There are only n-1 weight matrices, so we do the last case separately.
        n = self.sizes[-1]
        self.inputs.append(np.zeros((n,1),dtype=np.float32))
        self.outputs.append(np.zeros((n,1),dtype=np.float32))
        self.errors.append(np.zeros((n,1),dtype=np.float32))

    def build_network(self):
        #List of weight matrices taking the output of one layer to the input of the next.
        self.weights=[]
        #Bias vector for each layer.
        self.biases=[]
        self.weights_batch=[]
        #Bias vector for each layer.
        self.biases_batch=[]
        #Input vector for each layer.
        self.inputs=[]
        #Output vector for each layer.
        self.outputs=[]
        #Vector of errors at each layer.
        self.errors=[]
        #We initialise the weights randomly, and fill the other vectors with 1s.
        for layer in range(self.n_layers-1):
            n = self.sizes[layer]
            m = self.sizes[layer+1]
            self.weights.append(np.random.normal(0,1, (m,n)))
            self.biases.append(np.random.normal(0,1,(m,1)))
            self.weights_batch.append(np.random.normal(0,1, (m,n)))
            self.biases_batch.append(np.random.normal(0,1,(m,1)))
            self.inputs.append(np.zeros((n,1)))
            self.outputs.append(np.zeros((n,1)))
            self.errors.append(np.zeros((n,1)))
        #There are only n-1 weight matrices, so we do the last case separately.
        n = self.sizes[-1]
        self.inputs.append(np.zeros((n,1)))
        self.outputs.append(np.zeros((n,1)))
        self.errors.append(np.zeros((n,1)))

    def feedforward(self, x):
        #Propagates the input from the input layer to the output layer.
        k=len(x)
        x.shape=(k,1)
        self.inputs[0]=x
        self.outputs[0]=x
        for i in range(1,self.n_layers):
            self.inputs[i]=self.weights[i-1].dot(self.outputs[i-1])+self.biases[i-1]
            self.outputs[i]=self.fs[i](self.inputs[i])
        return self.outputs[-1]

    def feedforward_batch(self, X):
        #Propagates the input from the input layer to the output layer.

        self.inputs[0]=X
        self.outputs[0]=X
        for i in range(1,self.n_layers):
            self.inputs[i]=(self.weights[i-1].dot(self.outputs[i-1].T)+self.biases[i-1]).T
            self.outputs[i]=self.fs[i](self.inputs[i])

        return self.outputs[-1]

    def update_weights_batch(self, X, H1, H2, learning_rate=0.1):
        self.learning_rate=learning_rate
        delta=cca_prime(H1, H2)
        output = self.predict(X)
        self._zero_weights()
        for i in range(X.shape[0]):
            self._compute_weights_batchmode(X[i,:], delta[i:i+1,:])
        #self._update_weights()

    def _zero_weights(self):
        for i in range(len(self.weights)):
            self.weights_batch[i][:,:] = 0.0
            self.biases_batch[i][:] = 0.0

    def _update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i]-self.learning_rate*self.weights_batch[i]
            self.biases[i] = self.biases[i]-self.learning_rate*self.biases_batch[i]

    def update_weights_online(self,x, delta):
        #Update the weight matrices for each layer based on a single input x and target y.
        output = self.feedforward(x)
        self.errors[-1]=self.fprimes[-1](self.outputs[-1])*(delta.T)

        n=self.n_layers-2
        for i in xrange(n,0,-1):
            self.errors[i] = self.fprimes[i](self.inputs[i])*self.weights[i].T.dot(self.errors[i+1])
            self.weights[i] = self.weights[i]-self.learning_rate*np.outer(self.errors[i+1],self.outputs[i])
            self.biases[i] = self.biases[i] - self.learning_rate*self.errors[i+1]
        self.weights[0] = self.weights[0]-self.learning_rate*np.outer(self.errors[1],self.outputs[0])
        self.biases[0] = self.biases[0] - self.learning_rate*self.errors[1]

    def _compute_weights_batchmode(self,x, delta):
        #Update the weight matrices for each layer based on a single input x and target y.
        output = self.feedforward(x)
        self.errors[-1]=self.fprimes[-1](self.outputs[-1])*(delta.T)

        n=self.n_layers-2
        for i in xrange(n,0,-1):
            self.errors[i] = self.fprimes[i](self.inputs[i])*self.weights[i].T.dot(self.errors[i+1])
            self.weights_batch[i] += (np.outer(self.errors[i+1],self.outputs[i])+0.00001*self.weights[i])
            self.biases_batch[i] += (self.errors[i+1])+0.00001*self.biases[i]
        self.weights_batch[0] += (np.outer(self.errors[1],self.outputs[0])+0.00001*self.weights[0])
        self.biases_batch[0] += self.errors[1] + +0.00001*self.biases[0]

    def train(self,n_iter, learning_rate=1):
        #Updates the weights after comparing each input in X with y
        #repeats this process n_iter times.
        self.learning_rate=learning_rate
        n=self.X.shape[0]
        for repeat in range(n_iter):
            #We shuffle the order in which we go through the inputs on each iter.
            index=list(range(n))
            np.random.shuffle(index)
            for row in index:
                x=self.X[row]
                y=self.y[row]
                self.update_weights(x,y)

    def predict_x(self, x):
        return self.feedforward(x)

    def predict(self, X):
        n = len(X)
        m = self.sizes[-1]
        ret = np.ones((n,m),dtype=np.float32)
        for i in range(len(X)):
            ret[i,:] = self.feedforward(X[i])[:,0]
        return ret

class dCCA(object):
    def __init__(self, X1, X2, netCCA1, netCCA2):
        self.netCCA1 = netCCA1
        self.netCCA2 = netCCA2
        self.X1 = X1
        self.X2 = X2
        self.A1=np.eye(netCCA1.sizes[-1])
        self.A2=np.eye(netCCA1.sizes[-1])

    def predict_x1(self, X1):
        return self.netCCA1.predict(X1)
    def predict_x2(self, X2):
        return self.netCCA2.predict(X2)

    def train(self,n_iter, learning_rate=0.05):
        # Updates the weights after comparing each input in X with y
        # repeats this process n_iter times.
        self.learning_rate=learning_rate

        for repeat in range(n_iter):
            H1 = self.netCCA1.predict(self.X1)
            H2 = self.netCCA2.predict(self.X2)

            self.A1, self.A2, _H1, _H2 = CCA(H1,H2)
            # _H1 = np.dot(H1, self.A1)
            # _H2 = np.dot(H2, self.A2)
            print '--------------'
            print 'A1 :',A1.shape
            print 'H1 :',H1.shape
            print '_H1 :',_H1.shape
            print 'W :', self.netCCA1.weights[0].shape
            print '--------------'


            X1_rec = np.tanh(H1.dot(self.netCCA1.weights[0]))
            X2_rec = np.tanh(H2.dot(self.netCCA2.weights[0]))
            print repeat, 'mse1:', np.mean((X1_rec-self.X1)**2.0)
            print repeat, 'mse2:', np.mean((X2_rec-self.X2)**2.0)

            #if first:
            self.netCCA1.update_weights_batch(self.X1, H1, H2, self.learning_rate)
            self.netCCA1._update_weights()
            #else:
            self.netCCA2.update_weights_batch(self.X2, H2, H1, self.learning_rate)
            self.netCCA2._update_weights()
            #H1 = self.netCCA1.predict(self.X1)
            #H2 = self.netCCA2.predict(self.X2)
            #self.A1, self.A2, _H1, _H2 = CCA(H1,H2)
            #print repeat, cor_cost(_H1, _H2)

def test_regression():
    # Loading the data
    datasets = load_data_half('mnist.pkl.gz')

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    train_set_x = train_set_x.eval()
    train_set_y = train_set_y.eval()

    test_set_x = test_set_x.eval()
    test_set_y = test_set_y.eval()

    valid_set_x = valid_set_x.eval()
    valid_set_y = valid_set_y.eval()

    param1=((train_set_x.shape[1],0,0),(500, expit, logistic_prime),(50, expit, logistic_prime))
    param2=((train_set_y.shape[1],0,0),(500, expit, logistic_prime),(50, expit, logistic_prime))

    N1=netCCA(train_set_x,param1)
    N2=netCCA(train_set_y,param2)
    N = dCCA(train_set_x, train_set_y, N1, N2)


if __name__ == '__main__':
    test_regression()
