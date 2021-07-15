# %%
import matplotlib.pyplot as plt
import numpy as np

data = [(2.5, 2.4), (0.5, 0.7), (2.2, 2.9), (1.9, 2.2), (3.1, 3.0), (2.3, 2.7), (2.0, 1.6), (1.0, 1.1), (1.5, 1.6), (1.1, 0.9)]

def show(data):
    plt.figure(0)
    plt.title('original data')
    plt.axis([-3, 6, -2, 6])
    data = np.transpose(data)
    plt.scatter([i[0] for i in data], [i[1] for i in data])
    plt.show()

# PCA
x = np.transpose(np.array(data)) # (2,10) matrix
show(x)
x = x - np.mean(x, axis=1, keepdims=True) #center to mean
show(x)
C = (x @ np.transpose(x))/len(data) # covariant matrix
[u, s, v] = np.linalg.svd(C) # use singular value decomposition to calculate PCs (eigen vectors) 

#u has 2 eigen vector, transform x into a new space (eigen space) 
x1 = np.transpose(u) @ x
show(x1)

u = u[:,0:1] #reduce output dimension to one, select PC1
x2 = np.transpose(u) @ x #use u to reduce the data dimension 
x3 = u @ x2 #x2:1D data, project back to 2D space to visualize
show(x3)
