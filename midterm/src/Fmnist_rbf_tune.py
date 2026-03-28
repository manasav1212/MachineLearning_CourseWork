from models import *

path = "../data/Fashion-MNIST"
X_train, y_train, X_test, y_test = load_MINST_dataset(path)
X_train, X_test = flatten_images(X_train, X_test)

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np

def tune_rbf_pipeline(X_train, y_train, X_test, y_test, hp_list, dim_reduction=None):
    results = {}
    
    for hp in hp_list:
        if dim_reduction is None:
            pipeline = Pipeline([
                ('scaler', TimedTask(StandardScaler())),
                ('svc', TimedTask(SVC(kernel='rbf', C=hp[0], gamma=hp[1],  random_state=42)))
            ], verbose=True)
            print(f"Training RBF SVC with hp={hp}...")
            results[hp] = evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test)
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

fine_tuned_options = [
    (1, 0.001),
    (10, 0.001),
    (50, 0.001),
    (100, 0.001),
]
result = tune_rbf_pipeline(X_train, y_train, X_test, y_test, fine_tuned_options)
plot_rbf_bar_chart(result, "Without reduction")

fine_tuned_options_50 = [
    (1, 0.003),
    (10, 0.0005),
    (10, 0.001),
    (30, 0.001),
    (30, 0.003),
    (50, 0.003),
    (100, 0.001),  
]
result_50 = tune_rbf_pipeline(X_train, y_train, X_test, y_test, fine_tuned_options_50, PCA(n_components=50))
plot_rbf_bar_chart(result_50, "PCA-50")

fine_tuned_options_100 = [
    (10, 0.0001),
    (10, 0.0005),
    (10, 0.001),
    (30, 0.0002),
    (50, 0.0001),
    (100, 0.0003),
    (100, 0.0005),
    (300, 0.0001), 
]
result_100 = tune_rbf_pipeline(X_train, y_train, X_test, y_test, fine_tuned_options_100, PCA(n_components=100))
plot_rbf_bar_chart(result_100, "PCA-100")

fine_tuned_options_200 = [
    (30, 0.00005),
    (30, 0.00001),
    (50, 0.00001),
    (100, 0.0001),
    (100, 0.00005),
    (300, 0.0001),
    (300, 0.00005),
    (500, 0.00001),
]
result_200 = tune_rbf_pipeline(X_train, y_train, X_test, y_test, fine_tuned_options_200, PCA(n_components=200))
plot_rbf_bar_chart(result_200, "PCA-200")

fine_tuned_options_lda = [
    (1, 0.01),
    (1, 0.1),
    (10, 0.01),
    (10, 0.1),
    (10, 1),
    (50, 0.01),
    (50, 0.1),
    (50, 1),
]
result_lda = tune_rbf_pipeline(X_train, y_train, X_test, y_test, fine_tuned_options_lda, LinearDiscriminantAnalysis(n_components=9))
plot_rbf_bar_chart(result_lda, "LDA")


