#%%
import sys
sys.path.append('..')
import ml
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


data = tf.keras.datasets.mnist.load_data()

class DecisionTree:
    def __init__(self, data) -> None:
        # prepare data
        (x_train, y_train), (x_test, y_test) = data
        x_train = np.reshape(x_train, (-1, 28*28)) / 255 
        x_test = np.reshape(x_test, (-1, 28*28)) / 255
        self.y_train = y_train
        self.y_test = y_test

        # apply PCA to reduce dimension
        pca = ml.PCA()
        self.pca = pca
        pca.fit(x_train, compression_rate=0.99)
        self.x_train = pca.reduce_dim(x_train, use_whitening=False)
        self.x_test = pca.reduce_dim(x_test, use_whitening=False)

    def visualize(self):
        img = self.pca.recover_dim(self.x_train[3])
        img = np.reshape(img, (28,28))
        plt.imshow(img)

    def train(self):
        x, y, xt, yt = self.x_train, self.y_train, self.x_test, self.y_test
        
    
dt = DecisionTree(data)
# dt.visualize()



# %%
