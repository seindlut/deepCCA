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



with open('models/dae/f30_unc_test0.pkl', 'rb') as input:
    z = cPickle.load(input)
z = z.reshape((28,28))*255
x = SET[0,:].reshape((28,28))*255
recon = Image.fromarray(z)
original = Image.fromarray(x)
np.mean(z-x)
diff = Image.fromarray(z-x)
f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(10,4))
ax1.imshow(original)
ax1.axis('off')
ax1.set_title('original')
ax2.imshow(recon)
ax2.set_title('Reconstructed')
ax2.axis('off')
ax3.imshow(diff)
ax3.set_title('Difference')
ax3.axis('off')
plt.savefig('models/dae/illust.png',bbox_inches='tight')



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
