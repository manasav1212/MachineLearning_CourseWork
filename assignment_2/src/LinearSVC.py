import numpy as np

class LinearSVC:

    def __init__(self, eta = 0.01, epochs = 100, random_state=None, lam = 1.0):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.lam = lam

    def fit(self, X, y):
        n = X.shape[0]
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.losses_ = []

        for _ in range(self.epochs):
            y_cap = self.net_input(X)
            # Need the loss only for misclassified or inside the boundary classification
            y_mis_idx = y * y_cap < 1
            # Dot product gives the sum of product of yi * xi
            w_adj = self.eta / n * np.dot(X[y_mis_idx].T, y[y_mis_idx])
            self.w_ = self.w_ * (1 - self.eta * self.lam) + w_adj
            self.b_ = self.b_ + self.eta / n * np.where(y*y_cap < 1, y, 0).sum()  
        return self
    
    def net_input(self, X):
        return X@self.w_ + self.b_
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)