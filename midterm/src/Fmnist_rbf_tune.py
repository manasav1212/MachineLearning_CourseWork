from lib import *

path = "../data/Fashion-MNIST"
X_train, y_train, X_test, y_test = load_flattened_dataset(path)

fine_tuned_options = [
    (1, 0.001),
    (10, 0.001),
    (50, 0.001),
    (100, 0.001),
]
result = tune_rbf_pipeline(X_train, y_train, X_test, y_test, fine_tuned_options)
result, table = plot_rbf_bar_chart(result, "Without reduction")
print(result)
print(table)

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
result_50, table_50 = plot_rbf_bar_chart(result_50, "PCA-50")
print(result_50)
print(table_50)

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
result_100, table_100 = plot_rbf_bar_chart(result_100, "PCA-100")
print(result_100)
print(table_100)

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
result_200, table_200 = plot_rbf_bar_chart(result_200, "PCA-200")
print(result_200)
print(table_200)

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
result_lda, table_lda = plot_rbf_bar_chart(result_lda, "LDA")
print(result_lda)
print(table_lda)


