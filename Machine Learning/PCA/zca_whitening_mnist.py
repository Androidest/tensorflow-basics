#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(x, _), _ = tf.keras.datasets.mnist.load_data()

m = 5000
x = (np.reshape(x[:m,:,:], (m, 28*28)) / 255).T

def zca_whitening(x):
    x = x - np.mean(x, axis=1, keepdims=True) # Centralize data
    sigma = x @ x.T / m # calculate covariante matrix
    [u, s, _] = np.linalg.svd(sigma) # calculate PCs & eigen values
    W = u @ np.diag(1/np.sqrt(s)) # whitening: normalize each dim variation to 1, eigen values are PCs variations
    x1 = W.T @ x # transform the data to the new eigen space without data loss(no dimensional reduction) 
    x2 = u @ x1 # project back to the original space
    return x2

x_zca = zca_whitening(x)
plt.title('MNIST ZCA Whitening')
img = np.reshape(x_zca.T[3], (28,28))
plt.imshow(img)

# %%
