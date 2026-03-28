import numpy as np
from models import *
import time

# Load Dataset
path = "../data/MNIST"
X_train, y_train, X_test, y_test = load_MINST_dataset(path)
X_train, X_test = flatten_images(X_train, X_test)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

def create_pipeline(dim_reduction, model):
    return Pipeline([
        ('standardization', StandardScaler()),
        ('reduce_dim', dim_reduction),
        ('classifier', model)
    ], verbose=False)

def predict(models, X_test):
    combined_prediction = []
    for m in models:
        combined_prediction.append(m.predict(X_test))
    df = pd.DataFrame(combined_prediction)
    return df.mode().iloc[0].values

def predict_and_evaluate(models, X_test, y_test):
    y_pred = predict(models, X_test)
    return (y_pred == y_test).mean()

def split_data(num, X_train, y_train):
    total_size = X_train.shape[0]
    X_subset = []
    y_subset = []
    for i in range(num):
        X_subset.append(X_train[i * total_size//num: (i+1) * total_size//num ,])
        y_subset.append(y_train[i * total_size//num: (i+1) * total_size//num ,])
    return X_subset, y_subset

# For MNIST dataset
num_models = 9
X_subset, y_subset = split_data(9, X_train, y_train)

ensembled_models_mnist = [
    create_pipeline(PCA(n_components=50), SVC(kernel='linear', C = 0.01, random_state=7)),
    create_pipeline(PCA(n_components=100), SVC(kernel='linear', C = 0.01, random_state=7)),
    create_pipeline(PCA(n_components=100), SVC(kernel='linear', C = 0.01, random_state=7)),

    create_pipeline(PCA(n_components=50), SVC(kernel='rbf', C = 40, gamma=0.001, random_state=7)),
    create_pipeline(PCA(n_components=100), SVC(kernel='rbf', C = 40, gamma=0.001, random_state=7)),
    create_pipeline(PCA(n_components=200), SVC(kernel='rbf', C = 40, gamma=0.001, random_state=7)),

    create_pipeline(PCA(n_components=50), SVC(kernel='poly', C = 30 , gamma=0.001, degree = 3, random_state=7)),
    create_pipeline(PCA(n_components=100), SVC(kernel='poly', C = 30 , gamma=0.001, degree = 3, random_state=7)),
    create_pipeline(PCA(n_components=200), SVC(kernel='poly', C = 30 , gamma=0.001, degree = 3, random_state=7))
]
# Train all the models
start = time.perf_counter()
for i, model in enumerate(ensembled_models_mnist):
    model.fit(X_subset[i], y_subset[i])
training_time = time.perf_counter() - start
print(f"Training time: {training_time} s")
accuracy = predict_and_evaluate(ensembled_models_mnist, X_test, y_test)
print(f'Accuracy: {accuracy}, Error: {1 - accuracy}')


