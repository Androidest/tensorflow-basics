#%%
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# prepare data
(x,y), (tx,ty) = tf.keras.datasets.mnist.load_data()
x = np.reshape(x, (-1, 28*28))/255
tx = np.reshape(tx, (-1, 28*28))/255

# apply PCA
pca = PCA(n_components=3)
pca.fit(x)
x1 = pca.transform(x).T
# or: x1 = pca.fit_transform(x).T

plt.scatter(x1[0], x1[1], c=y, cmap="tab10")
plt.show()

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(x1[0], x1[1], x1[2], c=y, cmap='tab10')
plt.show()