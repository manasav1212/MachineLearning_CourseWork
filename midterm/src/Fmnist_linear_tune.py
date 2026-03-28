from lib import *

path = "../data/Fashion-MNIST"
X_train, y_train, X_test, y_test = load_flattened_dataset(path)

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


