import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

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

X_train, X_test, y_train, y_test, a = make_classification(d=2, n=100, u=10, seed=7)

clf_train = Perceptron().fit(X_train, y_train)
w_train = clf_train.coef_[0]
b_train = clf_train.intercept_[0]
x_vals_train = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
y_vals_train = -(w_train[0]/w_train[1])*x_vals_train-b_train/w_train[1]

plt.figure()
plt.scatter(X_train[y_train==-1, 0], X_train[y_train==-1, 1], color="red", label='Train: Class -1')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],color="blue", label='Train: Class 1')

plt.plot(x_vals_train, y_vals_train, "k--", label="Seperating Hyperplane")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linearly Separable Data: Train")
plt.legend()
plt.grid(True)
plt.show()

clf_test = Perceptron().fit(X_test, y_test)
w_test = clf_test.coef_[0]
b_test = clf_test.intercept_[0]
x_vals_test = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
y_vals_test = -(w_test[0]/w_test[1])*x_vals_test-b_test/w_test[1]
plt.figure()
plt.scatter(X_test[y_test==-1, 0], X_test[y_test==-1, 1], color="green", label='Test: Class -1')
plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], color="orange", label='Test: Class 1')
plt.plot(x_vals_test, y_vals_test, "k--", label="Seperating Hyperplane")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linearly Separable Data: Test")
plt.legend()
plt.grid(True)
plt.show()