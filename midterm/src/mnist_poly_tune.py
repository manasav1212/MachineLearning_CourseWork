from lib import *
import itertools

# Load Dataset
path = "../data/MNIST"
X_train, y_train, X_test, y_test = load_flattened_dataset(path)

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

options_50 = list(itertools.product(C, gamma, deg))
options_100 = list(itertools.product(C, gamma, deg))
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

result_50, table_50 = plot_poly_bar_chart(result_50, "PCA-50")
result_100, table_100 = plot_poly_bar_chart(result_100, "PCA-100")
result_200, table_200 = plot_poly_bar_chart(result_200, "PCA-200")

print(result_50)
print(table_50)
print(result_100)
print(table_100)
print(result_200)
print(table_200)

C_lda = [0.01, 0.1, 1, 10, 100]
gamma_lda = [0.001, 0.1, 1]
deg_lda = [2,3]
options_lda = list(itertools.product(C_lda, gamma_lda, deg_lda))
fine_options_lda = [(0.1, 0.1, 3),(1, 0.1, 3),(10, 0.1, 3),(0.01, 1, 3),(0.1, 1, 3),(0.5, 0.1, 3)]
result_lda = tune_poly_pipeline(X_train, y_train, X_test, y_test, fine_options_lda, LinearDiscriminantAnalysis(n_components=9))
result_lda, table_lda = plot_poly_bar_chart(result_lda, "LDA")
table_lda

C_all = [0.01, 0.1, 1, 10, 100]
gamma_all= [0.001, 0.1, 1]
deg_all = [2,3]
options_all = list(itertools.product(C_all, gamma_all, deg_all))
fine_options_all = [(1, 0.001, 2), (10, 0.001, 2),(100, 0.001, 2),(10, 0.01, 2),(10, 0.01, 3)]
result_all = tune_poly_pipeline(X_train, y_train, X_test, y_test, fine_options_all)
result_all, table_all = plot_poly_bar_chart(result_all, "Full features")
print(result_all)
print(table_all)


