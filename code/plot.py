import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pickle
# w/ Ipython kernel
%matplotlib inline


''' Uncorrupted '''

with open('models/dae/dAE_mnist_log.pkl', 'rb') as input:
    mse_log = pickle.load(input)

with open('models/dae/dAE_mnist_test_log.pkl', 'rb') as input:
    mse_log_test = pickle.load(input)

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
    mse_log = pickle.load(input)

plt.figure(figsize=(6,4))
plt.plot(range(len(mse_log)), mse_log)
plt.xlabel('Epoch')
plt.ylabel('MSE reconstruction')
plt.title('Deep Denoising Autoencoder (1 hidden layer)')
plt.savefig('models/dae/MSE_dae_corr30.png',bbox_inches='tight')



''''
SdAE
''''


with open('models/sdae/SdAE_mnist_pre_log.pkl', 'rb') as input:
    mse_layer = pickle.load(input)

plt.figure(figsize=(6,4))
for i in range(len(mse_layer)):
    plt.plot(range(len(mse_layer{i})), mse_layer{i}, label='layer '+str(i))

plt.xlabel('Epoch')
plt.ylabel('MSE reconstruction - pretraining')
plt.title('SDAE pre-training')
plt.savefig('models/dae/sdae_pre.png',bbox_inches='tight')
