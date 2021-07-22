#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(x, _), _ = tf.keras.datasets.mnist.load_data()

m = 5000
x = (np.reshape(x[:m,:,:], (m, 28*28)) / 255).T
x = x - np.mean(x, axis=1, keepdims=True)
sigma = x @ x.T / m

[u, s, _] = np.linalg.svd(sigma)
u = u[:,:2]# get just PC1 & PC2

x_2d = u.T @ x

plt.title('MNIST PCA Visualization')
plt.scatter(x_2d[0], x_2d[1])

# %%
