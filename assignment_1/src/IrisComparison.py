import pandas as pd
import numpy as np
from AdalineGD import AdalineGD
from LogisticRegressionGD import LogisticRegressionGD
import matplotlib.pyplot as plt

def transform(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X-mean)/std


df_iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, encoding='utf-8')

y_iris = df_iris.iloc[0:100, 4].values
y_iris = np.where(y_iris == 'Iris-setosa', 0, 1)
x_iris = df_iris.iloc[0:100, [0,2]].values

x_iris_std = transform(x_iris)

n_iter = 50
eta = 0.01

a_iris = AdalineGD(eta = eta, n_iter = n_iter)
a_iris.fit(x_iris_std, y_iris)

lr_iris = LogisticRegressionGD(eta = eta, n_iter = n_iter)
lr_iris.fit(x_iris_std, y_iris)

plt.figure()
plt.plot(range(1, n_iter+1), a_iris.losses_, marker='o', label = 'Adaline')
plt.title("Loss Convergence on Iris")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(range(1, n_iter+1), lr_iris.losses_, marker='o', label = 'Logistic Regression')
plt.title("Loss Convergence on Iris")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()