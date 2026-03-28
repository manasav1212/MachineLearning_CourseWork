from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline
from lib import *

# Experiments
path = "../data/MNIST"
X_train, y_train, X_test, y_test = load_flattened_dataset(path) 

# Tuning without dimensionality reduction
# Coarse tuning
# C_all = [0.001, 0.01, 0.1, 1, 10]
# Fine tuning
C_all_fine = [0.001, 0.002, 0.003, 0.005, 0.007, 0.1]
results_all = tune_linear_pipeline(X_train, y_train, X_test, y_test, C_all_fine)
results_all, table_all = plot_pipeline_results(results_all, "Full features")
print(results_all)
print(table_all)

# Tuning for PCA-50
c_options = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 1, 10]
results_50 = tune_linear_pipeline(X_train, y_train, X_test, y_test, c_options, PCA(n_components=50))
results_50, table_50 = plot_pipeline_results(results_50, "PCA-50")
print(results_50)
print(table_50)

# Tuning for PCA-100
results_100 = tune_linear_pipeline(X_train, y_train, X_test, y_test, c_options, PCA(n_components=100))
results_100, table_100 = plot_pipeline_results(results_100, "PCA-100")
print(results_100)
print(table_100)

# Tuning for PCA-200
c_options_200 = [0.0001, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.1, 1]
results_200 = tune_linear_pipeline(X_train, y_train, X_test, y_test, c_options_200, PCA(n_components=200))
results_200, table_200 = plot_pipeline_results(results_200, "PCA-200")
print(results_200)
print(table_200)

# Tuing for LDA
C_lda = [0.001, 0.01, 0.1, 0.5, 1, 3, 5, 10]
results_lda = tune_linear_pipeline(X_train, y_train, X_test, y_test, C_lda, LinearDiscriminantAnalysis(n_components=9))
results_lda, table_lda = plot_pipeline_results(results_lda, "LDA")
print(results_lda)
print(table_lda)