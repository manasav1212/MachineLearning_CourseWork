import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from AdalineGD import AdalineGD
from LogisticRegressionGD import LogisticRegressionGD
import matplotlib.pyplot as plt

df_iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, encoding='utf-8')
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None, encoding='utf-8')

y_iris = df_iris.iloc[0:100, 4].values
y_iris = np.where(y_iris == 'Iris-setosa', 0, 1)
x_iris = df_iris.iloc[0:100, [0,2]].values

df_wine = df_wine[df_wine.iloc[:,0].isin([1,2])]
y_wine = df_wine.iloc[:, 0].values
y_wine = np.where(y_wine == 1, 0, 1)
x_wine = df_wine.iloc[:, 1:].values

stdsc = StandardScaler()
x_iris_std = stdsc.fit_transform(x_iris)
x_wine_std = stdsc.fit_transform(x_wine)

n_iter = 50
eta = 0.01

a_iris = AdalineGD(eta = eta, n_iter = n_iter)
a_iris.fit(x_iris_std, y_iris)
a_wine = AdalineGD(eta = eta, n_iter = n_iter)
a_wine.fit(x_wine_std, y_wine)

lr_iris = LogisticRegressionGD(eta = eta, n_iter = n_iter)
lr_iris.fit(x_iris_std, y_iris)
lr_wine = LogisticRegressionGD(eta = eta, n_iter = n_iter)
lr_wine.fit(x_wine_std, y_wine)

plt.figure()
plt.plot(range(1, n_iter+1), a_iris.losses_, marker='o', label = 'Adaline')
plt.plot(range(1, n_iter+1), lr_iris.losses_, marker='x', label = 'Logistic Regression')
plt.title("Loss Convergence on Iris")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(range(1, n_iter+1), a_wine.losses_, marker='o', label = 'Adaline')
plt.plot(range(1, n_iter+1), lr_wine.losses_, marker='x', label = 'Logistic Regression')
plt.title("Loss Convergence on Wine")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()