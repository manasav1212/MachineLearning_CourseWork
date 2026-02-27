import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
plt.figure()
plt.scatter(X_train[y_train==-1, 0], X_train[y_train==-1, 1], color="red", label='Train: Class -1')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],color="blue", label='Train: Class 1')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linearly Separable Data: Train")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(X_test[y_test==-1, 0], X_test[y_test==-1, 1], color="green", label='Test: Class -1')
plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], color="orange", label='Test: Class 1')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linearly Separable Data: Test")
plt.legend()
plt.grid(True)
plt.show()