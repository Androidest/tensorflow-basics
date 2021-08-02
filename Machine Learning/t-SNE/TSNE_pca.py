#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

m = 10000
# prepare data
(x, y), _ = tf.keras.datasets.mnist.load_data()
x = np.reshape(x[:m], (-1, 28*28)) / 255 - 0.5
y = y[:m]

# apply PCA to reduce dimension
pca = PCA(n_components=50)
x = pca.fit_transform(x)

# apply t-SNE to reduce more dimesion
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=400)
x = tsne.fit_transform(x).T

plt.scatter(x[0], x[1], c=y, cmap='tab10')

