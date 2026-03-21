
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def time_fxn(iter, fxn, *args, **kwargs):
    start = time.perf_counter()
    result = None
    for _ in range(iter):
        result = fxn(*args, **kwargs)
    t = (time.perf_counter() - start)/iter
    return result, t

def apply_pca(X_train, X_test, n_components):
    pca = PCA(n_components=n_components)
    X_train_red = pca.fit_transform(X_train)
    X_test_red = pca.transform(X_test)
    return X_train_red, X_test_red, pca


def apply_lda(X_train, X_test, y_train, n_components):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_train_red = lda.fit_transform(X_train, y_train)
    X_test_red = lda.transform(X_test)
    return X_train_red, X_test_red, lda

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

def evaluate_model(model : BaseSvc, X_train, y_train, X_test, y_test):
    train_time = model.train(X_train, y_train)
    
    train_acc, train_cm = model.evaluate(X_train, y_train)
    test_acc, test_cm = model.evaluate(X_test, y_test)
    
    return {
        "train_time": train_time,
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        # "train_cm": train_cm,
        # "test_cm": test_cm
    }

def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)

    return {
        "train_time": train_time,
        "train_acc": train_acc,
        "test_acc": test_acc
    }