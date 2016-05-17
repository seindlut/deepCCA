import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pickle
# w/ Ipython kernel
%matplotlib inline


with open('models/dae/dAE_mnist_log.pkl', 'rb') as input:
    mse_log = pickle.load(input)

plt.figure(figsize=(6,4))
plt.plot(range(len(mse_log)), mse_log)
plt.xlabel('Epoch')
plt.ylabel('MSE reconstruction')
plt.title('Deep Autoencoder (1 hidden layer)')
plt.savefig('models/dae/MSE_dae.png',bbox_inches='tight')


with open('models/dae/dAE_mnist_corr30_log.pkl', 'rb') as input:
    mse_log = pickle.load(input)

plt.figure(figsize=(6,4))
plt.plot(range(len(mse_log)), mse_log)
plt.xlabel('Epoch')
plt.ylabel('MSE reconstruction')
plt.title('Deep Denoising Autoencoder (1 hidden layer)')
plt.savefig('models/dae/MSE_dae_corr30.png',bbox_inches='tight')
