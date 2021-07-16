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
def calcK(s):
    # one eigen value correspond to the total variation (sum of squared proj dist) of a PC
    totalEig = np.sum(s)
    sumEig = 0
    for k in range(len(s)):
        sumEig += s[k]
        if sumEig/totalEig > 0.999:
            return k
k = calcK(s)
print('Original dimension =', 28*28, ' k =',k)

u = u[:,:k] # select k principle component, k dimensions
y = u @ u.T @ x # reduce dimension from 784 to k & project back to 784

plt.title('MNIST PCA Reducing & compressing')
img = np.reshape(y.T[0], (28,28))
plt.imshow(img)

# %%
