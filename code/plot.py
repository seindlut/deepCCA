import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from six.moves import cPickle
from PIL import Image
import numpy as np
from logistic_sgd import load_data
# w/ Ipython kernel
%matplotlib inline

# Test set:
datasets = load_data('mnist.pkl.gz')
test_set_x, _ = datasets[2]
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
print 'final test: ',mse_log_test[-1]
print 'final train: ',mse_log[-1]

'''  Corruption 0.3 '''

with open('models/dae/dAE_mnist_corr30_log.pkl', 'rb') as input:
    mse_log = cPickle.load(input)

plt.figure(figsize=(6,4))
plt.plot(range(len(mse_log)), mse_log)
plt.xlabel('Epoch')
plt.ylabel('MSE reconstruction')
plt.title('Deep Denoising Autoencoder (1 hidden layer)')
plt.savefig('models/dae/MSE_dae_corr30.png',bbox_inches='tight')
print 'final train: ',mse_log[-1]



with open('models/dae/dAE_test_0.pkl', 'rb') as input:
    z = cPickle.load(input)
z = z.reshape((28,28))*255
x = SET[0,:].reshape((28,28))*255
recon = Image.fromarray(z)
original = Image.fromarray(x)
np.mean(z-x)
diff = Image.fromarray(z-x)
plt.imshow(diff)
plt.imshow(original)

plt.imshow(recon)


''''
SdAE
''''
with open('models/sdae/SdAE_fn_losses.pkl', 'rb') as input:
    fn = cPickle.load(input)
fntr = fn["train"]
fnt = fn["test"]
fnv = fn["val"]
corr = fn['corruption_levels']
with open('models/sdae/SdAE_mnist_pre_log.pkl', 'rb') as input:
    mse_layer = cPickle.load(input)
print len(mse_layer)
plt.figure(figsize=(6,4))
for i in range(len(mse_layer)):
    plt.plot(range(len(mse_layer[i])), mse_layer[i], label='layer '+str(i)+ '(noise = )'+str(corr[i]))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MSE reconstruction - pretraining')
plt.title('SDAE pre-training')
plt.savefig('models/dae/sdae_pre.png',bbox_inches='tight')


plt.plot(range(len(fntr)), fntr, label ='train')
plt.plot(range(len(fnv)), fnv, label = 'val')
plt.plot(range(len(fnt)), fnt, label = 'test')

plt.legend()
plt.xlabel("Finetuning epoch")
plt.ylabel('Error')
plt.title('Finetuning  - logistic regression')


""" Reconstruction """

with open('models/sdae/SdAE_test_recon-pretrain.pkl', 'rb') as input:
    Q = cPickle.load(input)

z = Q[0,:].reshape((28,28))*255
x = SET[0,:].reshape((28,28))*255
np.mean(z-x)
recon = Image.fromarray(z)
original = Image.fromarray(x)
np.mean(z-x)
diff = Image.fromarray(z-x)
plt.imshow(original)
plt.title('original')
plt.imshow(recon)
plt.title('Rec')
