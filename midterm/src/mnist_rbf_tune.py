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

import itertools
X_tune, _, y_tune, _ = train_test_split(
            X_train, 
            y_train, 
            train_size=10000, 
            stratify=y_train,
            random_state=42
        )

# C = [0.01, 1, 10, 100]
# gamma = [0.001, 0.01, 0.1]
C_100 = [30, 40, 50, 60, 80]
gamma_100 = [0.0001, 0.001]

C_200 = [1, 5, 10, 20]
gamma_200 = [0.01, 0.001, 0.0001]

options_100 = list(itertools.product(C_100, gamma_100))
options_200 = list(itertools.product(C_200, gamma_200))
fine_tuned_options_50 = [
    (10, 0.001),
    (100, 0.001),  
    (10, 0.0005),  
    (10, 0.005),   
    (5, 0.001),    
    (20, 0.001),
    (50, 0.001),  
    (100, 0.0005), 
]
fine_tuned_options_100 = list(itertools.product([30, 40, 50],[0.0001, 0.001]))
fine_tuned_options_200 = [
    (5, 0.001),
    (20, 0.001),
    (30, 0.001),
    (40, 0.001),
    (50, 0.001),
]
result_50 = tune_rbf_pipeline(X_train, y_train, X_test, y_test, fine_tuned_options_50, PCA(n_components=50))
result_100 = tune_rbf_pipeline(X_train, y_train, X_test, y_test, fine_tuned_options_100, PCA(n_components=100))
result_200 = tune_rbf_pipeline(X_train, y_train, X_test, y_test, fine_tuned_options_200, PCA(n_components=200))

plot_rbf_bar_chart(result_50, "PCA-50")
plot_rbf_bar_chart(result_100, "PCA-100")
plot_rbf_bar_chart(result_200, "PCA-200")

C_all = [0.01, 1, 10, 100]
gamma_all = [0.0001, 0.001, 0.01, 0.1]
option_fine_all = [(0.01, 0.001),(1, 0.001),(10, 0.001),(100, 0.001)]

options_all = list(itertools.product(C_all, gamma_all))
result_all = tune_rbf_pipeline(X_train, y_train, X_test, y_test, option_fine_all)
result_all,table_all = plot_rbf_bar_chart(result_all, "All features")

C_lda = [0.01, 1, 10, 100]
gamma_lda = [0.0001, 0.001, 0.01, 0.1]
options_lda = list(itertools.product(C_lda, gamma_lda))
option_fine_lda = [(1, 0.1),(10, 0.01),(5, 0.05),(1, 0.05), (100, 0.01)]
result_lda = tune_rbf_pipeline(X_train, y_train, X_test, y_test, option_fine_lda, LinearDiscriminantAnalysis(n_components=9))
result_lda,table_lda = plot_rbf_bar_chart(result_lda, "LDA")
print(result_lda)
print(table_lda)


