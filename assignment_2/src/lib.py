import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product
import pandas as pd
import os

def make_classification(d, n, u, seed):
    #Step 1. Randomly generate a d-dimensional vector ¯a.
    rng = np.random.default_rng(seed)
    a = rng.normal(size=d)
    
    #Step 2. Randomly select n samples ¯x1, . . . , ¯xn in the range of [−u, u] in each dimension. 
    # You may use a uniform or Gaussian distribution to do so.
    X = rng.uniform(-u, u, size=(n, d))
    
    #Step 3. Give each ¯xi a label yi such that if ¯aT ¯x < 0 then yi = −1, otherwise yi = 1.
    y = np.where(X @ a < 0, -1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed)
    return X_train, X_test, y_train, y_test, a


def generate_and_save_dataset(dimensions, size):

    for n,d in product(dimensions, size):
        X_train, X_test, y_train, y_test, a = make_classification(d, n, 100, 7)
        df_train = pd.DataFrame(X_train)
        df_train['label'] = y_train

        df_test = pd.DataFrame(X_test)
        df_test['label'] = y_test

        df_train.to_csv(f'../data/train_{n}_{d}.csv')
        df_test.to_csv(f'../data/test_{n}_{d}.csv')


def load_dataset(dimension, size):
    
    train_path = f"../data/train_{dimension}_{size}.csv"
    test_path = f"../data/test_{dimension}_{size}.csv"
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
        
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    df_train = pd.read_csv(train_path, index_col=0)
    df_test = pd.read_csv(test_path, index_col=0)
    
    X_train = df_train.drop(columns=["label"]).values
    y_train = df_train["label"].values
    
    X_test = df_test.drop(columns=["label"]).values
    y_test = df_test["label"].values
    
    return X_train, X_test, y_train, y_test

class LinearSVC:

    def __init__(self, eta = 0.01, epochs = 100, random_state=None, lam = 1):
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
            self.w_ = self.w_ * (1 - self.eta * self.lam / n) + w_adj
            self.b_ = self.b_ + self.eta / n * np.where(y*y_cap < 1, y, 0).sum()
            # Save the losses
            y_cap_new = self.net_input(X)
            loss = np.mean(np.maximum(0, 1 - y * y_cap_new)) + self.lam / (2 * n) * np.dot(self.w_, self.w_)
            self.losses_.append(loss)
        return self
    
    def net_input(self, X):
        return X@self.w_ + self.b_
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)