
""" Linear algebra tools """

import numpy as np

from numpy.linalg import inv, lstsq, cholesky
from scipy.linalg import sqrtm

try:
    from matplotlib import pyplot as pp
    import matplotlib.cm as cm

except ImportError:
    print 'matplotlib is could not be imported'
import sklearn.cross_decomposition
#from sklearn.cross_decomposition import CCA as CCA
from cca_linear import cca as CCA
import warnings
warnings.simplefilter("ignore")
def order_cost(H1,H2):
    #_cca=CCA(n_components=H1.shape[1])
    #x,y=_cca.fit_transform(H1, H2)
    a,b,x,y=CCA(H1,H2)
    return cor_cost(x,y), cca(x,y)
def mat_pow(matrix):
    return sqrtm(inv(matrix))
def stable_inverse_Aneg1_dot_B(A,B):
    # solves x=A^-1.B or A.x=B
    return lstsq(A,B)[0]

def solve_cholesky(A,B):
    L=cholesky(A)
    #assert np.allclose(np.dot(L, L.T),A)
    y=lstsq(L,B)[0]
    return lstsq(L.T,y)
def stable_inverse_Aneg1_dot_B_cholesky(A,B):
    # solves x=A^-1.B or A.x=Bm where A=L.L.T
    ##L=cholesky(A)
    ##assert np.allclose(np.dot(L, L.T),A)
    ##y=lstsq(L,B)[0]
    return solve_cholesky(A,B)[0]
    ##return lstsq(L.T,y)[0]
def stable_inverse_A_dot_Bneg1(A,B):
    # solves x=A.B^-1 or x.A=B
    return lstsq(B, A.T)[0].T
def stable_inverse_A_dot_Bneg1_cholesky(A,B):
    # solves x=A.B^-1 or x.A=B
    return solve_cholesky(B, A.T)[0].T
def mat_pow2(matrix):
    return inv(sqrtm(matrix))

    #return matrix ** -0.5
from mlp_numpy import *
from SdA_mapping import load_data_half, plot_weights
def cor_cost(H1,H2):
    cor=0.0
    for i in range(H1.shape[1]):
        cur_cor = abs(np.corrcoef(H1[:,i], H2[:,i])[0,1])
        if not np.isnan(cur_cor):
            cor += cur_cor
        else:
            cor += 1.0
        #if np.corrcoef(H1[:,i], H2[:,i])[0,1] < 0:
        #    print 'negative'
    return cor
def cca_cost(H1, H2):
    return (cca(H1, H2)+cca(H2, H1))/(cca(H1, H1)+cca(H2, H2))
def cca(H1, H2):
    H1bar = copy.deepcopy(H1)
    H1bar = H1bar-H1bar.mean(axis=0)
    H2bar = copy.deepcopy(H2)
    H2bar = H2bar-H2bar.mean(axis=0)
    H1bar = H1bar.T
    H2bar = H2bar.T
    H1bar += np.random.random(H1bar.shape)*0.00001
    H2bar += np.random.random(H2bar.shape)*0.00001
    r1 = 0.00000001
    m = H1.shape[0]
    #H1bar = H1 - (1.0/m)*np.dot(H1, np.ones((m,m), dtype=np.float32))
    #H2bar = H2 - (1.0/m)*np.dot(H2, np.ones((m,m), dtype=np.float32))
    SigmaHat12 = (1.0/(m-1))*np.dot(H1bar, H2bar.T)
    SigmaHat11 = (1.0/(m-1))*np.dot(H1bar, H1bar.T)
    SigmaHat11 = SigmaHat11 + r1*np.identity(SigmaHat11.shape[0], dtype=np.float32)
    SigmaHat22 = (1.0/(m-1))*np.dot(H2bar, H2bar.T)
    SigmaHat22 = SigmaHat22 + r1*np.identity(SigmaHat22.shape[0], dtype=np.float32)
    SigmaHat11_2=mat_pow(SigmaHat11).real.astype(np.float32)
    SigmaHat22_2=mat_pow(SigmaHat22).real.astype(np.float32)
    ##TMP = np.dot(SigmaHat12, SigmaHat22_2) #unstable
    TMP2 = stable_inverse_A_dot_Bneg1(SigmaHat12, sqrtm(SigmaHat22))#np.dot(SigmaHat12, SigmaHat22_2)
    TMP3 = stable_inverse_A_dot_Bneg1_cholesky(SigmaHat12, sqrtm(SigmaHat22))#np.dot(SigmaHat12, SigmaHat22_2)

    ##Tval = np.dot(SigmaHat11_2, TMP) #unstable
    Tval = stable_inverse_Aneg1_dot_B(sqrtm(SigmaHat11), TMP2)
    Tval3 = stable_inverse_Aneg1_dot_B_cholesky(sqrtm(SigmaHat11), TMP3)

    ##U, D, V, = np.linalg.svd(Tval)

    ## corr =  np.trace(np.dot(Tval.T, Tval))**(0.5) #wrong
    corr =  np.trace(sqrtm(np.dot(Tval.T, Tval)))
    return corr

def cca_prime(H1, H2):
    H1bar = copy.deepcopy(H1)
    H1bar = H1bar-H1bar.mean(axis=0)
    H2bar = copy.deepcopy(H2)
    H2bar = H2bar-H2bar.mean(axis=0)
    H1bar = H1bar.T
    H2bar = H2bar.T
    H1bar += np.random.random(H1bar.shape)*0.00001
    H2bar += np.random.random(H2bar.shape)*0.00001
    r1 = 0.00000001
    m = H1.shape[0]
    #H1bar = H1 - (1.0/m)*np.dot(H1, np.ones((m,m), dtype=np.float32))
    #H2bar = H2 - (1.0/m)*np.dot(H2, np.ones((m,m), dtype=np.float32))
    SigmaHat12 = (1.0/(m-1))*np.dot(H1bar, H2bar.T)
    SigmaHat11 = (1.0/(m-1))*np.dot(H1bar, H1bar.T)
    SigmaHat11 = SigmaHat11 + r1*np.identity(SigmaHat11.shape[0], dtype=np.float32)
    SigmaHat22 = (1.0/(m-1))*np.dot(H2bar, H2bar.T)
    SigmaHat22 = SigmaHat22 + r1*np.identity(SigmaHat22.shape[0], dtype=np.float32)
    SigmaHat11_2=mat_pow(SigmaHat11).real.astype(np.float32)
    SigmaHat22_2=mat_pow(SigmaHat22).real.astype(np.float32)
    ##TMP = np.dot(SigmaHat12, SigmaHat22_2) #unstable
    TMP3 = stable_inverse_A_dot_Bneg1_cholesky(SigmaHat12, sqrtm(SigmaHat22))#np.dot(SigmaHat12, SigmaHat22_2)

    ##Tval = np.dot(SigmaHat11_2, TMP) #unstable
    Tval = stable_inverse_Aneg1_dot_B_cholesky(sqrtm(SigmaHat11), TMP3)
    U, D, V, = np.linalg.svd(Tval)
    D=np.diag(D)
    UVT = np.dot(U, V)
    Delta12 = np.dot(SigmaHat11_2, np.dot(UVT, SigmaHat22_2))
    UDUT = np.dot(U, np.dot(D, U.T))
    Delta11 = (-0.5) * np.dot(SigmaHat11_2, np.dot(UDUT, SigmaHat11_2))
    grad_E_to_o = (1.0/m) * (2*np.dot(Delta11,H1bar)+np.dot(Delta12,H2bar))
    ##gparam1_W = (grad_E_to_o) * (h1tmpval*(1-h1tmpval)) * (h1hidden)
    #gparam1_W = -1.0*numpy.dot((h1hidden), ((grad_E_to_o) * (h1tmpval*(1-h1tmpval))).T)
    ##gparam1_b = (grad_E_to_o) * (h1tmpval*(1-h1tmpval)) * theano.shared(numpy.array([1.0],dtype=theano.config.floatX), borrow=True)
    #gparam1_b = -1.0*numpy.dot(numpy.ones((1,10000),dtype=theano.config.floatX), ((grad_E_to_o) * (h1tmpval*(1-h1tmpval))).T)
    #gparam1_W = theano.shared(gparam1_W, borrow=True)
    #gparam1_b = theano.shared(gparam1_b[0,:], borrow=True)
    return -1.0*grad_E_to_o.T##np.real(grad_E_to_o.real).T

class netCCA_old(object):
    def __init__(self, X, parameters):
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
        self.build_network()

    def build_network(self):
        #List of weight matrices taking the output of one layer to the input of the next.
        self.weights=[]
        #Bias vector for each layer.
        self.biases=[]
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
        #k=len(x)
        #x.shape=(k,1)
        self.inputs[0]=x
        self.outputs[0]=x
        for i in range(1,self.n_layers):
            ##self.inputs[i]=self.weights[i-1].dot(self.outputs[i-1])+self.biases[i-1]
            self.inputs[i]=np.dot(self.outputs[i-1], self.weights[i-1].T)+self.biases[i-1].T
            self.outputs[i]=self.fs[i](self.inputs[i])
        return self.outputs[-1]
    def update_weights(self,X,H2,learning_rate=0.1):
        #Update the weight matrices for each layer based on a single input x and target y.
        self.learning_rate=learning_rate
        output = self.predict(X)
        self.errors[-1]=self.fprimes[-1](self.outputs[-1])*cca_prime(output, H2)

        n=self.n_layers-2
        for i in xrange(n,0,-1):
            self.errors[i] = self.fprimes[i](self.inputs[i])*np.dot(self.errors[i+1], self.weights[i])
            ##self.weights[i] = self.weights[i]-self.learning_rate*#np.outer(self.errors[i+1],self.outputs[i])
            self.weights[i] = self.weights[i]-self.learning_rate*np.dot(self.errors[i+1].T, self.outputs[i])#np.outer(self.errors[i+1],self.outputs[i])
            ##self.biases[i] = self.biases[i] - self.learning_rate*self.errors[i+1]
            self.biases[i] = self.biases[i] - self.learning_rate*np.dot(self.errors[i+1].T, np.ones((self.outputs[i].shape[0],1)))

        ##self.weights[0] = self.weights[0]-self.learning_rate*np.outer(self.errors[1],self.outputs[0])
        self.weights[0] = self.weights[0]-self.learning_rate*np.dot(self.errors[1].T, self.outputs[0])
        ##self.biases[0] = self.biases[0] - self.learning_rate*self.errors[1]
        self.biases[0] = self.biases[0] - self.learning_rate*np.dot(self.errors[1].T, np.ones((self.outputs[0].shape[0],1)))

    def train(self,n_iter, learning_rate=1):
        #Updates the weights after comparing each input in X with y
        #repeats this process n_iter times.
        self.learning_rate=learning_rate
        n=self.X.shape[0]
        for repeat in range(n_iter):
            print repeat
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
        #n = len(X)
        #m = self.sizes[-1]
        #ret = np.ones((n,m))
        #for i in range(len(X)):
        #    ret[i,:] = self.feedforward(X[i])
        return self.feedforward(X)



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

class netCCA_nobias(object):
    def __init__(self, X, parameters,Ws=None):
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
        if Ws is None:
            self.build_network()
        else:
            self.import_weights(Ws)

    def import_weights(self, Ws):
        #List of weight matrices taking the output of one layer to the input of the next.
        self.weights=[]
        #Bias vector for each layer.
        self.weights_batch=[]
        self.weights_rec_batch=[]

        #Input vector for each layer.
        self.inputs=[]
        self.inputs_rec=[]

        #Output vector for each layer.
        self.outputs=[]
        self.outputs_rec=[]

        #Vector of errors at each layer.
        self.errors=[]
        self.errors_rec=[]

        #We initialise the weights randomly, and fill the other vectors with 1s.
        for layer in range(self.n_layers-1):
            n = self.sizes[layer]
            m = self.sizes[layer+1]
            self.weights.append(Ws[layer])
            self.weights_batch.append(np.random.normal(0,1, (m,n)))
            self.weights_rec_batch.append(np.random.normal(0,1, (self.sizes[-layer-1-1],self.sizes[-layer-1])))
            self.inputs.append(np.zeros((n,1),dtype=np.float32))
            self.outputs.append(np.zeros((n,1),dtype=np.float32))
            self.inputs_rec.append(np.zeros((self.sizes[-layer-1],1),dtype=np.float32))
            self.outputs_rec.append(np.zeros((self.sizes[-layer-1],1),dtype=np.float32))
            self.errors.append(np.zeros((n,1),dtype=np.float32))
            self.errors_rec.append(np.zeros((self.sizes[-layer-1],1),dtype=np.float32))

        #There are only n-1 weight matrices, so we do the last case separately.
        n = self.sizes[-1]
        self.inputs.append(np.zeros((n,1),dtype=np.float32))
        self.outputs.append(np.zeros((n,1),dtype=np.float32))
        self.errors.append(np.zeros((n,1),dtype=np.float32))
        self.inputs_rec.append(np.zeros((self.sizes[0],1),dtype=np.float32))
        self.outputs_rec.append(np.zeros((self.sizes[0],1),dtype=np.float32))
        self.errors_rec.append(np.zeros((self.sizes[0],1),dtype=np.float32))

    def build_network(self):
        #List of weight matrices taking the output of one layer to the input of the next.
        self.weights=[]
        self.weights_batch=[]
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
            self.weights_batch.append(np.random.normal(0,1, (m,n)))
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
            self.inputs[i]=self.weights[i-1].dot(self.outputs[i-1])
            self.outputs[i]=self.fs[i](self.inputs[i])
        self.inputs_rec[0] = self.outputs[i]
        self.outputs_rec[0] = self.outputs[i]
        for i in range(self.n_layers-1, 0,-1):
            self.inputs_rec[i]=self.weights[i-1].T.dot(self.outputs_rec[i-1])
            self.outputs_rec[i]=self.fs[i](self.inputs_rec[i])
        return self.outputs[-1], self.outputs_rec[-1]
    def feedforward_batch(self, X):
        #Propagates the input from the input layer to the output layer.

        self.inputs[0]=X
        self.outputs[0]=X
        for i in range(1,self.n_layers):
            self.inputs[i]=(self.weights[i-1].dot(self.outputs[i-1].T)).T
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
        a=0
    def _zero_weights(self):
        for i in range(len(self.weights)):
            self.weights_batch[i][:,:] = 0.0
    def _update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i]-self.learning_rate*self.weights_batch[i]*(1.0/50000.0)##
    def update_weights_online(self,x, delta):
        #Update the weight matrices for each layer based on a single input x and target y.
        output = self.feedforward(x)
        self.errors[-1]=self.fprimes[-1](self.outputs[-1])*(delta.T)

        n=self.n_layers-2
        for i in xrange(n,0,-1):
            self.errors[i] = self.fprimes[i](self.inputs[i])*self.weights[i].T.dot(self.errors[i+1])
            self.weights[i] = self.weights[i]-self.learning_rate*np.outer(self.errors[i+1],self.outputs[i])
        self.weights[0] = self.weights[0]-self.learning_rate*np.outer(self.errors[1],self.outputs[0])
    def _compute_weights_batchmode(self,x, delta):
        #Update the weight matrices for each layer based on a single input x and target y.
        output, rec = self.feedforward(x)

        # Rec gradient dec
        self.errors_rec[-1]=self.fprimes[-1](self.outputs_rec[-1])*(rec-x)
        n=self.n_layers-2
        for i in xrange(n,0,-1):
            self.errors_rec[i] = self.fprimes[i](self.inputs_rec[i])*self.weights[i].dot(self.errors_rec[i+1])
            self.weights_rec_batch[i] += (np.outer(self.errors_rec[i+1],self.outputs_rec[i]))
        self.errors_rec[0] = self.weights[0].dot(self.errors_rec[1])
        self.weights_rec_batch[0] = (np.outer(self.errors_rec[1],self.outputs_rec[0]))

        # DCCA gradient
        ##self.errors[-1]=self.fprimes[-1](self.outputs[-1])*(delta.T+self.errors_rec[0])
        std_delta = (self.fprimes[-1](self.outputs[-1])*(delta.T)).std()+1e-06
        std_err = self.errors_rec[0].std()+1e-06
        coef = 1.0#std_err/std_delta
        self.errors[-1]=(self.fprimes[-1](self.outputs[-1])*(delta.T))*coef##+(self.errors_rec[0])
        ##self.errors[-1]=self.errors_rec[0]
        n=self.n_layers-2
        for i in xrange(n,0,-1):
            self.errors[i] = self.fprimes[i](self.inputs[i])*self.weights[i].T.dot(self.errors[i+1])
            self.weights_batch[i] += (np.outer(self.errors[i+1],self.outputs[i]))
        std_w_err = self.weights_rec_batch[0].T.std()+1e-06
        std_w_delta = (np.outer(self.errors[1],self.outputs[0])).std()+1e-06
        coef = std_w_err/std_w_delta

        ##self.weights_batch[0] += (np.outer(self.errors[1],self.outputs[0])*coef)+self.weights_rec_batch[0].T
        self.weights_batch[0] += (np.outer(self.errors[1],self.outputs[0]))#+self.weights_rec_batch[0].T

        ##self.weights_batch[0] += (np.outer(self.errors[1],self.outputs[0]))+self.weights_rec_batch[0].T

    def reconstruct(self, x):
        k=len(x)
        x.shape=(k,1)
        inp=x
        out=x
        for i in range(1,self.n_layers):
            inp=self.weights[i-1].dot(out)
            out=self.fs[i](inp)
        inp = out
        out = out
        for i in range(self.n_layers-1, 0,-1):
            inp=self.weights[i-1].T.dot(out)
            out=self.fs[i](inp)
        return out

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
            ret[i,:] = self.feedforward(X[i])[0][:,0]
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
        #Updates the weights after comparing each input in X with y
        #repeats this process n_iter times.
        self.learning_rate=learning_rate
        #H1 = self.netCCA1.predict(self.X1[:,:])
        #H2 = self.netCCA2.predict(self.X2[:,:])
        #cca_prime(H1,H2)

        for repeat in range(n_iter):
            #We shuffle the order in which we go through the inputs on each iter.
            #index=list(range(n))
            #np.random.shuffle(index)
            #for row in index:
            #x=self.X[row]
            #y=self.y[row]
            #H1 = self.netCCA1.predict(self.X1)
            #H2 = self.netCCA2.predict(self.X2)
            #st = 0
            #en = min(10000, self.X1.shape[0])
            #cnt = 0
            H1 = self.netCCA1.predict(self.X1)
            H2 = self.netCCA2.predict(self.X2)
            #self.A1, self.A2, _H1, _H2 = CCA(H1,H2)
            #_H1 = np.dot(H1, self.A1)
            #_H2 = np.dot(H2, self.A2)

            self.A1, self.A2, _H1, _H2 = CCA(H1,H2)

            print repeat, cor_cost(_H1, _H2)

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


#expit is a fast way to compute logistic using precomputed exp.
from scipy.special import expit
def test_regression(plots=False):
    if 1:
        np.random.seed(0)
        X1=np.random.random((1000,50))
        X2=np.random.random((1000,50))
        X2[:,25:] = X1[:,:25]
        print cor_cost(X1,X2)
        cca(X1,X2)
    #First create the data.
    n=200
    X=np.linspace(0,3*np.pi,num=n)
    X.shape=(n,1)
    #y1=np.sin(X)
    #y2=np.sin(X+0.8*np.pi)
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
    y1 = test_set_x
    y2 = test_set_y

    A=np.random.random((10000,50))
    B=np.random.random((10000,50))
    #cca_prime(A, B)
    #cca_prime(A, A)
    #We make a neural net with 2 hidden layers, 20 neurons in each, using logistic activation
    #functions.
    param1=((y1.shape[1],0,0),(2038, expit, logistic_prime),(50, expit, logistic_prime))
    param2=((y2.shape[1],0,0),(1608, expit, logistic_prime),(50, expit, logistic_prime))
    np.random.seed(0)
    N1=netCCA(y1,param1)
    N2=netCCA(y2,param2)
    N = dCCA(train_set_x, train_set_y, N1, N2)
    if 0:
        net=NeuralNetwork(y1, y1, param1)
        for k in range(1000):
            net.train(test_set_x[:,:], test_set_x[:,:], 1, learning_rate=0.05)
            out=net.predict(valid_set_x)
            #print 'accuracy', k, ':', np.sum(np.argmax(out,1)==valid_label)
            print 'reconstruction', k, ':', np.mean(np.sum((out-valid_set_x)**2,1))
    #print 'accuracy', np.sum(np.argmax(out,1)==test_label)
    #plot_weights(net.weights[0])
    #out=net.predict(test_set_x)
    #Set learning rate.
    rates=[0.01]
    predictions=[]
    for rate in rates:
        N.train(10, learning_rate=rate)
        #predictions.append([rate,N.predict(X)])
    plot_weights(net.weights[0])

    import matplotlib.pyplot as pp
    h1=N.netCCA1.predict(y1)
    h2=N.netCCA2.predict(y2)
    ax=pp.subplot(211)
    pp.imshow(h1.T,interpolation='none',aspect='auto')
    pp.subplot(212,sharex=ax)
    pp.imshow(h2.T,interpolation='none',aspect='auto')
    pp.show()


    fig, ax=plt.subplots(1,1)
    if plots:
        ax.plot(X,y, label='Sine', linewidth=2, color='black')
        for data in predictions:
            ax.plot(X,data[1],label="Learning Rate: "+str(data[0]))
        ax.legend()
    #plt.show()
def plot_weights2(w):
    import matplotlib.pyplot as pp
    #ax=pp.subplot(211)
    #pp.imshow(w[0,:].reshape((28,28)),interpolation='none',aspect='auto')
    #pp.show()
    a=np.zeros((28*10,28*10))
    for i in range(100):
        m=i%10
        n=i/10
        a[m*28:(m+1)*28, n*28:(n+1)*28] = w[i,:].reshape((28,28))
    pp.imshow(a,interpolation='none',aspect='auto')
    pp.show()

def load_data(dataset='mnist.pkl.gz'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    import cPickle
    import os
    import sys
    import time
    import gzip

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
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    test_set_x, test_label = test_set
    valid_set_x, valid_label = valid_set
    train_set_x, train_label = train_set
    def get_onehot(data_y):
        data_y_new = np.zeros((data_y.shape[0], data_y.max()+1))
        for i in range(data_y.shape[0]):
            data_y_new[i, data_y[i]] = 1
        return data_y_new
    train_set_y = get_onehot(train_label)
    test_set_y = get_onehot(test_label)
    valid_set_y = get_onehot(valid_label)

    rval = [(train_set_x, train_set_y, train_label), (valid_set_x, valid_set_y, valid_label),
            (test_set_x, test_set_y, test_label)]
    return rval



if __name__ == '__main__':
    test_regression(True)
