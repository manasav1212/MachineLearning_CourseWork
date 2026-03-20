
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

import idx2numpy
import time

def load_MINST_dataset(path):
    train_data = idx2numpy.convert_from_file(f"{path}/train-images-idx3-ubyte")
    train_label = idx2numpy.convert_from_file(f"{path}/train-labels-idx1-ubyte")
    test_data = idx2numpy.convert_from_file(f"{path}/t10k-images-idx3-ubyte")
    test_label = idx2numpy.convert_from_file(f"{path}/t10k-labels-idx1-ubyte")
    return train_data, train_label, test_data, test_label

def flatten_images(train_data, test_data):
    return train_data.reshape(train_data.shape[0], -1), test_data.reshape(test_data.shape[0], -1)

def time_fxn(iter, fxn, *args, **kwargs):
    start = time.perf_counter()
    result = None
    for _ in range(iter):
        result = fxn(*args, **kwargs)
    t = (time.perf_counter() - start)/iter
    return result, t

class BaseSvc:

    def train(self, X, y):
        _, train_time = time_fxn(1, self.model.fit, X, y)
        return train_time
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        accuracy = (y_true == y_pred).mean()
        cm = confusion_matrix(y_true, y_pred)
        return accuracy, cm

class LinearSvc(BaseSvc):

    def __init__(self, C, max_iter=1000, random_state = 42):
        super().__init__()
        self.C = C
        self.model = SVC(kernel='linear', C = self.C, max_iter = max_iter, random_state = random_state)

class RbfSvc(BaseSvc):

    def __init__(self, C, gamma = 'scale', max_iter=1000, random_state = 42):
        super().__init__()
        self.C = C
        self.gamma = gamma
        self.model = SVC(kernel='rbf', C = self.C, gamma= self.gamma, max_iter = max_iter, random_state= random_state)

class PolynomialSvc(BaseSvc):

    def __init__(self, C, degree, gamma = 'scale', max_iter=1000, random_state = 42):
        super().__init__()
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.model = SVC(kernel='poly', C = self.C, gamma= self.gamma, max_iter = max_iter, random_state= random_state,
                          degree= self.degree)