#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

(x, y), _ = tf.keras.datasets.mnist.load_data()

m = 5000
x = np.reshape(x[:m,:,:], (m, 28*28)).T / 255
x = x - np.mean(x, axis=1, keepdims=True)
y = y[:m]

# PCA
sigma = x @ x.T / m
[u, s, _] = np.linalg.svd(sigma)
u = u[:,:3]# get just PC1 & PC2

# 2D
plt.title('MNIST PCA 2D Visualization')
x_2d = u.T @ x # project data to the new feature space
plt.scatter(x_2d[0], x_2d[1], c=y, cmap='tab10') # tab10 means 10 colors
plt.show()

# 3D
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
x_3d = u.T @ x # project data to the new feature space
ax.scatter(x_3d[0], x_3d[1], x_3d[2], c=y, cmap='tab10')
plt.show()


