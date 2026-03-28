from models import *

# Load Dataset
path = "../data/MNIST"
X_train, y_train, X_test, y_test = load_MINST_dataset(path)
X_train, X_test = flatten_images(X_train, X_test)

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np

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

import itertools
X_tune, _, y_tune, _ = train_test_split(
            X_train, 
            y_train, 
            train_size=20000, 
            stratify=y_train,
            random_state=42
        )

C = [0.1, 1, 5, 30, 50]
gamma = [0.001, 0.05,  0.01]
deg = [2,3]

options_50 = list(itertools.product(C,gamma, deg))
options_100 = list(itertools.product(C,gamma, deg))
fine_options_50 = [
    (0.1, 0.01, 3), #This was the best one yet 
    (0.1, 0.01, 2),
    (5, 0.001, 2),
    (1, 0.01, 3),    
    (30, 0.001, 2),
    (30, 0.001, 3),  
    (50, 0.001, 2),
    (50, 0.001, 3)
]

fine_options_100 = [
    (0.1, 0.01, 3), # The best one on sub-samples
    (0.1, 0.01, 2),
    (5, 0.001, 2),
    (30, 0.001, 2),
    (30, 0.001, 3),
    (50, 0.001, 2),
    (50, 0.001, 3)
]

C_200 = [0.01, 0.1, 1, 30, 50]
gamma_200 = [ 0.01, 0.001]
deg_200 = [2,3]
options_200 = list(itertools.product(C_200,gamma_200, deg_200))
fine_options_200 = [
    (0.1, 0.01, 3), # The best one on sub-samples
    (0.1, 0.01, 2),
    (30, 0.001, 2),
    (30, 0.001, 3),
    (50, 0.001, 3),
]

result_50 = tune_poly_pipeline(X_train, y_train, X_test, y_test, fine_options_50, PCA(n_components=50))
result_100 = tune_poly_pipeline(X_train, y_train, X_test, y_test, fine_options_100, PCA(n_components=100))
result_200 = tune_poly_pipeline(X_train, y_train, X_test, y_test, fine_options_200, PCA(n_components=200))

plot_poly_bar_chart(result_50, "PCA-50")
plot_poly_bar_chart(result_100, "PCA-100")
plot_poly_bar_chart(result_200, "PCA-200")

C_lda = [0.01, 0.1, 1, 10, 100]
gamma_lda = [0.001, 0.1, 1]
deg_lda = [2,3]
options_lda = list(itertools.product(C_lda, gamma_lda, deg_lda))
fine_options_lda = [(0.1, 0.1, 3),(1, 0.1, 3),(10, 0.1, 3),(0.01, 1, 3),(0.1, 1, 3),(0.5, 0.1, 3) ]
result_lda = tune_poly_pipeline(X_train, y_train, X_test, y_test, fine_options_lda, LinearDiscriminantAnalysis(n_components=9))
result_lda, table_lda = plot_poly_bar_chart(result_lda, "LDA")
table_lda

import itertools
X_tune, _, y_tune, _ = train_test_split(
            X_train, 
            y_train, 
            train_size=10000, 
            stratify=y_train,
            random_state=42
        )
C_all = [0.01, 0.1, 1, 10, 100]
gamma_all= [0.001, 0.1, 1]
deg_all = [2,3]
options_all = list(itertools.product(C_all, gamma_all, deg_all))
fine_options_all = [(1, 0.001, 2), (10, 0.001, 2),(100, 0.001, 2),(10, 0.01, 2),(10, 0.01, 3)]
result_all = tune_poly_pipeline(X_train, y_train, X_test, y_test, fine_options_all)
result_all, table_all = plot_poly_bar_chart(result_all, "Full features")
print(result_all)
print(table_all)


