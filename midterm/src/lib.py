
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import idx2numpy
import time

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

def load_MINST_dataset(path):
    train_data = idx2numpy.convert_from_file(f"{path}/train-images-idx3-ubyte")
    train_label = idx2numpy.convert_from_file(f"{path}/train-labels-idx1-ubyte")
    test_data = idx2numpy.convert_from_file(f"{path}/t10k-images-idx3-ubyte")
    test_label = idx2numpy.convert_from_file(f"{path}/t10k-labels-idx1-ubyte")
    return train_data, train_label, test_data, test_label

def flatten_images(train_data, test_data):
    return train_data.reshape(train_data.shape[0], -1), test_data.reshape(test_data.shape[0], -1)

def load_flattened_dataset(path):
    train_data, train_label, test_data, test_label = load_MINST_dataset(path)
    train_data, test_data = flatten_images(train_data, test_data)
    return train_data, train_label, test_data, test_label

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

    step_times = {}
    for name, step in pipeline.named_steps.items():
        if hasattr(step, "training_time"):
            step_times[name] = step.training_time

    return {
        "train_time": train_time,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "step_times": step_times
    }

from sklearn.base import BaseEstimator, TransformerMixin

# Wrapper to save training time of each pipeline steps
class TimedTask(BaseEstimator, TransformerMixin):

    def __init__(self, task, name=None):
        self.task = task
        self.name = name

    def fit(self, X, y=None):
        start = time.perf_counter()
        # Need this for internal fit tracking
        self.model_ = self.task
        self.model_.fit(X, y)
        self.training_time = time.perf_counter() - start
        return self

    def transform(self, X):
        return self.model_.transform(X)

    def fit_transform(self, X, y=None):
        start = time.perf_counter()
        # Need this for internal fit tracking
        self.model_ = self.task
        m = self.model_.fit_transform(X, y)
        self.training_time = time.perf_counter() - start
        return m

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y):
        return self.model_.score(X, y)

def tune_linear_pipeline(X_train, y_train, X_test, y_test, c_options, dim_reduction = None):
    results = {}
    
    for c in c_options:
        if dim_reduction is None:
            pipeline = Pipeline([
                ('scaler', TimedTask(StandardScaler())),
                ('svc', TimedTask(SVC(kernel='linear', C=c, random_state=42)))
            ], verbose=True)
        else:
            pipeline = Pipeline([
                ('scaler', TimedTask(StandardScaler())),
                ('reduce_dim', TimedTask(dim_reduction)),
                ('svc', TimedTask(SVC(kernel='linear', C=c, random_state=42)))
            ], verbose=True)
        print(f"Training Linear SVC with C={c}...")
        results[c] = evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test)
        
    return results

def plot_pipeline_results(result, title):
    x = list(result.keys())
    train_time = [result[k]['train_time'] for k in x]
    train_acc = [result[k]['train_acc'] for k in x]
    test_acc = [result[k]['test_acc'] for k in x]

    plt.figure(figsize=(6.5, 4.5))

    # Accuracy
    plt.plot(x, train_acc, marker='o', label='Train Accuracy', color='blue')
    plt.plot(x, test_acc, marker='s', label='Test Accuracy', color='green')
    plt.xscale('log')
    plt.xlabel('Hyperparameter (C)')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Test Accuracy ({title})')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    table = pd.DataFrame(result).T
    table = table[['train_time', 'train_acc', 'test_acc', 'step_times']].copy()
    table.columns = ['Training Time','Training Accuracy', 'Test Accuracy', 'Step Times']

    return result, table

def tune_poly_pipeline(X_train, y_train, X_test, y_test, hp_list, dim_reduction = None):
    results = {}
    
    for hp in hp_list:
        if (dim_reduction is None):
            pipeline = Pipeline([
                ('scaler', TimedTask(StandardScaler())),
                ('svc', TimedTask(SVC(kernel='poly', C=hp[0], gamma=hp[1], degree = hp[2], random_state=42)))
            ], verbose=True)
        else:
            pipeline = Pipeline([
                ('scaler', TimedTask(StandardScaler())),
                ('reduce_dim', TimedTask(dim_reduction)),
                ('svc', TimedTask(SVC(kernel='poly', C=hp[0], gamma=hp[1], degree = hp[2], random_state=42)))
            ], verbose=True)
        print(f"Training Polynomial SVC with hp={hp}...")
        results[hp] = evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test)
        
    return results

def plot_poly_bar_chart(result, title):
    tuples = list(result.keys())
    
    x_labels = [f"C={c}\nγ={g}\nd={d}" for c, g, d in tuples]

    train_acc = [result[k]['train_acc'] for k in tuples]
    test_acc = [result[k]['test_acc'] for k in tuples]

    x = np.arange(len(x_labels)) 
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, train_acc, width, label='Train Accuracy', color='#4C72B0')
    ax.bar(x + width/2, test_acc, width, label='Test Accuracy', color='#55A868')

    ax.set_ylabel('Accuracy')
    ax.set_title(f'Hyperparameter Tuning: {title}')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    min_acc = min(min(train_acc), min(test_acc))
    ax.set_ylim([max(0, min_acc - 0.02), 1.0])

    plt.tight_layout()
    plt.show()

    table = pd.DataFrame(result).T
    table.index = pd.MultiIndex.from_tuples(table.index, names=['C', 'Gamma', 'Degree'])
    table = table[['train_time', 'train_acc', 'test_acc', 'step_times']].copy()
    table.columns = ['Training Time','Training Accuracy', 'Test Accuracy', 'Step Times']

    return result, table

def tune_rbf_pipeline(X_train, y_train, X_test, y_test, hp_list, dim_reduction = None):
    results = {}
    
    for hp in hp_list:
        if dim_reduction is None:
            pipeline = Pipeline([
                ('scaler', TimedTask(StandardScaler())),
                ('svc', TimedTask(SVC(kernel='rbf', C=hp[0], gamma=hp[1],  random_state=42)))
            ], verbose=True)
        else:
            pipeline = Pipeline([
                ('scaler', TimedTask(StandardScaler())),
                ('reduce_dim', TimedTask(dim_reduction)),
                ('svc', TimedTask(SVC(kernel='rbf', C=hp[0], gamma=hp[1],  random_state=42)))
            ], verbose=True)
        print(f"Training RBF SVC with hp={hp}...")
        results[hp] = evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test)
        
    return results

def plot_rbf_bar_chart(result, title):
    tuples = list(result.keys())
    x_labels = [f"C={c}\nγ={g}" for c, g in tuples]

    train_acc = [result[k]['train_acc'] for k in tuples]
    test_acc = [result[k]['test_acc'] for k in tuples]

    x = np.arange(len(x_labels)) 
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, train_acc, width, label='Train Accuracy', color='#4C72B0')
    ax.bar(x + width/2, test_acc, width, label='Test Accuracy', color='#55A868')

    ax.set_ylabel('Accuracy')
    ax.set_title(f'Hyperparameter Tuning: {title}')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    min_acc = min(min(train_acc), min(test_acc))
    ax.set_ylim([max(0, min_acc - 0.02), 1.0])

    plt.tight_layout()
    plt.show()

    table = pd.DataFrame(result).T
    table.index = pd.MultiIndex.from_tuples(table.index, names=['C', 'Gamma'])
    table = table[['train_time', 'train_acc', 'test_acc', 'step_times']].copy()
    table.columns = ['Training Time','Training Accuracy', 'Test Accuracy', 'Step Times']

    return result, table
