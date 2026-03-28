from models import *

path = "../data/Fashion-MNIST"
X_train, y_train, X_test, y_test = load_MINST_dataset(path)
X_train, X_test = flatten_images(X_train, X_test)

from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd

def tune_linear_pipeline(X_train, y_train, X_test, y_test, c_options, dim_reduction=None):
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

c_options = [0.001, 0.01, 0.1, 1]
results = tune_linear_pipeline(X_train, y_train, X_test, y_test, c_options)
results, table = plot_pipeline_results(results, "Without Reduction")
print(results)
print(table)

c_options_50 = [0.0001, 0.001, 0.01, 0.1, 0.8, 1, 5, 10]
results_50 = tune_linear_pipeline(X_train, y_train, X_test, y_test, c_options_50, PCA(n_components=50))
results_50, table_50 = plot_pipeline_results(results_50, "PCA-50")
print(results_50)
print(table_50)

c_options_100 = [0.0001, 0.001, 0.01, 0.05, 0.5, 1]
results_100 = tune_linear_pipeline(X_train, y_train, X_test, y_test, c_options_100, PCA(n_components=100))
results_100, table_100 = plot_pipeline_results(results_100, "PCA-100")
print(results_100)
print(table_100)

c_options_200 = [0.0001, 0.001, 0.01, 0.1, 1, 10]
results_200 = tune_linear_pipeline(X_train, y_train, X_test, y_test, c_options_200, PCA(n_components=200))
results_200, table_200 = plot_pipeline_results(results_200, "PCA-200")
print(results_200)
print(table_200)

c_options_lda = [0.0001, 0.001, 0.01, 0.1, 1, 10]
results_lda = tune_linear_pipeline(X_train, y_train, X_test, y_test, c_options_lda, LinearDiscriminantAnalysis(n_components=9))
results_lda, table_lda = plot_pipeline_results(results_lda, "LDA")
print(results_lda)
print(table_lda)


