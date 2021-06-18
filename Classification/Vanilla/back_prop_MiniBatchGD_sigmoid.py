
#%%
import numpy as np
import matplotlib as plt
import tensorflow as tf # to get the MNIST dataset

# load data 
batch_size = 60
learning_rate = 0.001

(X,I), (TX, TY) = tf.keras.datasets.mnist.load_data()
m, h, w = X.shape
# Flatten images into vectors
X = np.reshape(X, (m//batch_size, batch_size, w*h)) / 255.0
TX = np.reshape(TX, (-1, w*h)) / 255.0
# turn labels into one hot form
Y = np.zeros(shape=(m, 10))
Y[np.arange(m), I] = 1
Y = np.reshape(Y, (m//batch_size, batch_size, 10)) 

# initialize output layers
W = np.zeros(shape=(28*28, 10))
b = np.zeros(shape=(10))

for epoch in range(60):
    for i in range(len(X)):
        batch_x = X[i]
        batch_y = Y[i]

        y = batch_x @ W + b
        sigmoid = 1.0 / (1.0 + np.exp(-y))

        # calculate cross-entropy -> softmax gradient
        dy = sigmoid - batch_y
        assert dy.shape == (batch_size, 10)
        dW = np.transpose(batch_x) @ dy / batch_size
        assert dW.shape == W.shape
        db = np.sum(dy, axis=0) / batch_size
        assert db.shape == b.shape

        # update parameters according to gradients
        W  -= learning_rate*dW
        b  -= learning_rate*db

    y = np.argmax(TX @ W + b, axis=1)
    accuracy = np.mean(y == TY)
    print(accuracy, flush=True)


    
    

    

# %%
