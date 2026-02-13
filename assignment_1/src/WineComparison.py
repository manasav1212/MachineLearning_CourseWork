import pandas as pd
import numpy as np
from AdalineGD import AdalineGD
from LogisticRegressionGD import LogisticRegressionGD
import matplotlib.pyplot as plt

def transform(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X-mean)/std


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None, encoding='utf-8')

df_wine = df_wine[df_wine.iloc[:,0].isin([1,2])]
y_wine = df_wine.iloc[:, 0].values
y_wine = np.where(y_wine == 1, 0, 1)
x_wine = df_wine.iloc[:, 1:].values

x_wine_std = transform(x_wine)

n_iter = 50
eta = 0.01

a_wine = AdalineGD(eta = eta, n_iter = n_iter)
a_wine.fit(x_wine_std, y_wine)

lr_wine = LogisticRegressionGD(eta = eta, n_iter = n_iter)
lr_wine.fit(x_wine_std, y_wine)

plt.figure()
plt.plot(range(1, n_iter+1), a_wine.losses_, marker='o', label = 'Adaline')
plt.title("Loss Convergence on wine")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(range(1, n_iter+1), lr_wine.losses_, marker='o', label = 'Logistic Regression')
plt.title("Loss Convergence on wine")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()