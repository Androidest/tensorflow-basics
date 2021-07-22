#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(x, y), _ = tf.keras.datasets.mnist.load_data()

m = 5000
x = np.reshape(x[:m,:,:], (m, 28*28)).T / 255
x = x - np.mean(x, axis=1, keepdims=True)
y = y[:m]

# PCA
sigma = x @ x.T / m
[u, s, _] = np.linalg.svd(sigma)
u = u[:,:2]# get just PC1 & PC2

# 
plt.title('MNIST PCA Visualization')
for i in range(10):
    indexes = np.argwhere(y == i)[:,0]
    x1 = x.T[indexes].T # select data corresponding to a single digit through label 
    x_2d = u.T @ x1 # project data to the new feature space
    plt.scatter(x_2d[0], x_2d[1])

plt.show()



