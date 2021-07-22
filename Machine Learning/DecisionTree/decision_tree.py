#%%
import sys
sys.path.append('..')
import ml
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class DecisionTree:
    def __init__(self, data) -> None:
        # prepare data
        (x_train, y_train), (x_test, y_test) = data
        x_train = np.reshape(x_train, (-1, 28*28))[:10000] / 255 - 0.5
        x_test = np.reshape(x_test, (-1, 28*28)) / 255 - 0.5
        self.x = x_train
        self.y_train = y_train[:10000]
        self.y_test = y_test

        # apply PCA to reduce dimension
        # pca = ml.PCA()
        # self.pca = pca
        # pca.fit(x_train, compression_rate=0.9)
        # self.x_train = pca.reduce_dim(x_train, use_whitening=False)
        # self.x_test = pca.reduce_dim(x_test, use_whitening=False)
        self.x_train = x_train
        self.x_test = x_test

    def visualize_img(self):
        img = self.pca.recover_dim(self.x_train[0])
        img = np.reshape(img, (28,28))
        plt.imshow(img)

    def visualize_2d(self):
        x1, x2 = self.pca.visualize_2d(self.x)
        plt.scatter(x1, x2)
        plt.show()

    def recursiveTrain(self, x, y, parentImpurity, classIndex):
        if len(y) == 0:
            return -1

        (bestDim, (lx, ly), (rx, ry), giniImpurity, gini_left, gini_right, left_class, right_class) = self.findBestDim(x, y)
        if giniImpurity >= parentImpurity:
            return classIndex
        
        return { 'dim':bestDim,
                 'left':self.recursiveTrain(lx, ly, gini_left, left_class), 
                 'right':self.recursiveTrain(rx, ry, gini_right, right_class) }

    def findBestDim(self, x, y):
        minImpurity = 1
        bestResult = None
        
        for i in range(len(x)):
            dim = x[i]
            result = self.node_gini(dim, y)
            if result[0] < minImpurity:
                minImpurity = result[0]
                bestResult = result + (i,)

        (gini, gini_left, gini_right, left_mask, right_mask, left_class, right_class, bestDim) = bestResult
        xt = x.T
        yt = y.T
        lx = xt[left_mask].T
        ly = yt[left_mask].T
        rx = xt[right_mask].T
        ry = yt[right_mask].T
        return (bestDim, (lx, ly), (rx, ry), gini, gini_left, gini_right, left_class, right_class)

    def branch_gini(self, labels):
        gini = 1
        m = len(labels)
        if m == 0:
            return gini, -1

        maxProb = 0
        maxProbClass = None

        for classIndex in range(10):
            p = np.count_nonzero(labels == classIndex) / m
            gini -= p*p

            if p > maxProb:
                maxProb = p
                maxProbClass = classIndex

        return gini, maxProbClass

    def node_gini(self, x, y):
        m = len(y)

        # True leaf gini impurity
        left_mask = np.argwhere(x >= 0)[:,0]
        gini_left, left_class = self.branch_gini(y[left_mask])    
        left_count = len(left_mask)        

        # False leaf gini impurity
        right_mask = np.argwhere(x < 0)[:,0]
        gini_right, right_class = self.branch_gini(y[right_mask])
        right_count = len(right_mask)
        
        # total gini impurity
        gini = (left_count * gini_left + right_count * gini_right) / m
        
        return (gini, gini_left, gini_right, left_mask, right_mask, left_class, right_class)

    def train(self):
        # dimension axis = 0, sample axis = 1
        x, y = self.x_train.T, self.y_train.T
        self.root = self.recursiveTrain(x, y, 1, 0)
    
    def test(self):
        m = len(self.x_test)
        count = 0

        for i in range(m):
            x = self.x_test[i]
            y = self.y_test[i]
            if self.predict(x) == y:
                count += 1
        
        print('Precision', count/m)

    def predict(self, x):
        node = self.root
        while(True): 
            if x[node['dim']] >= 0:
                node = node['left']
                if type(node) == int:
                    return node
            else:
                node = node['right']
                if type(node) == int:
                    return node


dt = DecisionTree(tf.keras.datasets.mnist.load_data())
# dt.visualize_2d()
# dt.visualize_img()
dt.train()
dt.test() #Precision 0.8624 without PCA


# %%
from sklearn import tree
import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 28*28))[:10000] / 255 - 0.5
x_test = np.reshape(x_test, (-1, 28*28)) / 255 - 0.5

dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train[:10000])
prediction = dt.predict(x_test)

accurancy = np.mean(np.equal(prediction, y_test), axis=-1)
print('accurancy:', accurancy) # accurancy: 0.8765
