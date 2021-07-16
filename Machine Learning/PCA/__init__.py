import numpy as np

class PCA:
    def __init__(self) -> None:
        self.k = None
        self.u = None
        self.s = None
        self.pc = None
        self.zca_pc = None

    def fit(self, x, compression_rate):
        x = x.T # turn x to default math form 
        n,m = x.shape # n dimension, m data samples
        x = x - np.mean(x, axis=1, keepdims=True) # centralize data
        sigma = x @ x.T / m 
        u, s, _ = np.linalg.svd(sigma)

        totalEig = np.sum(s)
        sumEig = 0
        k = 0    
        for k in range(len(s)):
            sumEig += s[k]
            if sumEig/totalEig > compression_rate:
                self.k = k
                self.u = u
                self.s = s
                self.pc = u[:,:k] # select the first kth PCs, for n to k dimensional reduction
                self.zca_pc = self.pc @ np.diag(1/np.sqrt(s[:k]))
                return
    
    def reduce_dim(self, x, use_whitening=False):
        x = x.T # turn x to default math form 
        if use_whitening:
            return (self.zca_pc.T @ x).T
        return (self.pc.T @ x).T

    def recover_dim(self, x):
        x = x.T # turn x to default math form 
        return (self.pc @ x).T