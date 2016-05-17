import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from six.moves import cPickle
from PIL import Image
import numpy as np
# w/ Ipython kernel
%matplotlib inline

# Test set:
datasets = load_data('mnist.pkl.gz')
test_set_x, test_set_y = datasets[1]
SET = test_set_x.get_value(borrow=True)




''' Uncorrupted '''

with open('models/dae/dAE_mnist_log.pkl', 'rb') as input:
    mse_log = cPickle.load(input)

with open('models/dae/dAE_mnist_test_log.pkl', 'rb') as input:
    mse_log_test = cPickle.load(input)

plt.figure(figsize=(6,4))
plt.plot(range(len(mse_log)), mse_log,label='train')
plt.plot(range(len(mse_log_test)), mse_log_test,label='test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MSE reconstruction')
plt.title('Deep Autoencoder (1 hidden layer)')
plt.savefig('models/dae/MSE_dae.png',bbox_inches='tight')


'''  Corruption 0.3 '''

with open('models/dae/dAE_mnist_corr30_log.pkl', 'rb') as input:
    mse_log = cPickle.load(input)

plt.figure(figsize=(6,4))
plt.plot(range(len(mse_log)), mse_log)
plt.xlabel('Epoch')
plt.ylabel('MSE reconstruction')
plt.title('Deep Denoising Autoencoder (1 hidden layer)')
plt.savefig('models/dae/MSE_dae_corr30.png',bbox_inches='tight')


with open('models/dae/dummy.pkl', 'rb') as input:
    z = cPickle.load(input)
z = z.reshape((28,28))*255
x = SET[0,:].reshape((28,28))*255
recon = Image.fromarray(z)
original = Image.fromarray(x)
np.mean(z-x)
diff = Image.fromarray(z-x)
plt.imshow(diff)
plt.imshow(original)
''''
SdAE
''''


with open('models/sdae/SdAE_mnist_pre_log.pkl', 'rb') as input:
    mse_layer = cPickle.load(input)

plt.figure(figsize=(6,4))
for i in range(len(mse_layer)):
    plt.plot(range(len(mse_layer{i})), mse_layer{i}, label='layer '+str(i))

plt.xlabel('Epoch')
plt.ylabel('MSE reconstruction - pretraining')
plt.title('SDAE pre-training')
plt.savefig('models/dae/sdae_pre.png',bbox_inches='tight')
