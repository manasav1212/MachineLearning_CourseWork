from lib import *

path = "../data/Fashion-MNIST"
X_train, y_train, X_test, y_test = load_flattened_dataset(path)

fine_options = [
    (10, 0.1, 2),
    (50, 0.001, 3),
    (100, 0.0001, 2),
    (300, 0.0001, 2),
]
result = tune_poly_pipeline(X_train, y_train, X_test, y_test, fine_options)
result, table = plot_poly_bar_chart(result, "Without reduction")
print(result)
print(table)

fine_options_50 = [
    (10, 0.001, 2),
    (10, 0.001, 3),
    (30, 0.001, 2),
    (30, 0.001, 3),
    (50, 0.001, 2),
    (50, 0.001, 3),
    (100, 0.001, 2),
    (100, 0.001, 3),
]
result_50 = tune_poly_pipeline(X_train, y_train, X_test, y_test, fine_options_50, PCA(n_components=50))
result_50, table_50 = plot_poly_bar_chart(result_50, "PCA-50")
print(result_50)
print(table_50)

fine_options_100 = [
    (10, 0.0005, 2),
    (10, 0.0005, 3),
    (30, 0.0005, 2),
    (30, 0.0003, 2),
    (30, 0.0005, 3),
    (100, 0.0005, 2),
    (100, 0.0003, 2),
    (100, 0.0005, 3),
]
result_100 = tune_poly_pipeline(X_train, y_train, X_test, y_test, fine_options_100, PCA(n_components=100))
result_100, table_100 = plot_poly_bar_chart(result_100, "PCA-100")
print(result_100)
print(table_100)

fine_options_200 = [
    (50, 0.0001, 2),
    (50, 0.00005, 2),
    (100, 0.0001, 2),
    (100, 0.00005, 2),
    (100, 0.0001, 3),
    (300, 0.0001, 2),
    (300, 0.00005, 2),
    (300, 0.0001, 3),
]
result_200 = tune_poly_pipeline(X_train, y_train, X_test, y_test, fine_options_200, PCA(n_components=200))
result_200, table_200 = plot_poly_bar_chart(result_200, "PCA-200")
print(result_200)
print(table_200)

fine_options_lda = [
    (10, 0.01, 2),
    (10, 0.01, 3),
    (10, 0.1, 2),
    (10, 0.1, 3),
    (50, 0.01, 2),
    (50, 0.01, 3),
    (50, 0.1, 2),
    (50, 0.1, 3),
]
result_lda = tune_poly_pipeline(X_train, y_train, X_test, y_test, fine_options_lda, LinearDiscriminantAnalysis(n_components=9))
result_lda, table_lda = plot_poly_bar_chart(result_lda, "LDA")
print(result_lda)
print(table_lda)


