# %%
# PCA:
# 1. Reduce memory/disk needed to store data
# 2. Speed up learning
# 3. Do not use it to prevent overfitting

import matplotlib.pyplot as plt
import numpy as np

data = [(2.5, 2.4), (0.5, 0.7), (2.2, 2.9), (1.9, 2.2), (3.1, 3.0), (2.3, 2.7), (2.0, 1.6), (1.0, 1.1), (1.5, 1.6), (1.1, 0.9)]

def show(data):
    plt.axis([-3, 6, -2, 6])
    plt.scatter(data[0,:], data[1,:])
    plt.show()

# PCA
m = len(data)
x = np.array(data).T # (2,10) matrix
show(x)
x = x - np.mean(x, axis=1, keepdims=True) #center to mean
show(x)
C = (x @ x.T)/m # covariant matrix
[u, s, v] = np.linalg.svd(C) # use singular value decomposition to calculate PCs (eigen vectors) 
# s are eigen_values, 是平均投影方差,因为前面的C除以样本数

#u has 2 eigen vector, transform x into a new space (eigen space) 
x1 = u.T @ x
show(x1)

u = u[:,0:1] #reduce output dimension to one, select PC1
x2 = u.T @ x #use u to reduce the data dimension 
x3 = u @ x2 #x2:1D data, project back to 2D space to visualize
show(x3)
