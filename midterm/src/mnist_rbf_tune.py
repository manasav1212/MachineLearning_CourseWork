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

result_50, table_50 = plot_rbf_bar_chart(result_50, "PCA-50")
result_100, table_100 = plot_rbf_bar_chart(result_100, "PCA-100")
result_200, table_200 = plot_rbf_bar_chart(result_200, "PCA-200")

print(result_50)
print(table_50)
print(result_100)
print(table_100)
print(result_200)
print(table_200)

C_all = [0.01, 1, 10, 100]
gamma_all = [0.0001, 0.001, 0.01, 0.1]
option_fine_all = [(0.01, 0.001),(1, 0.001),(10, 0.001),(100, 0.001)]

options_all = list(itertools.product(C_all, gamma_all))
result_all = tune_rbf_pipeline(X_train, y_train, X_test, y_test, option_fine_all)
result_all,table_all = plot_rbf_bar_chart(result_all, "All features")
print(result_all)
print(table_all)

C_lda = [0.01, 1, 10, 100]
gamma_lda = [0.0001, 0.001, 0.01, 0.1]
options_lda = list(itertools.product(C_lda, gamma_lda))
option_fine_lda = [(1, 0.1),(10, 0.01),(5, 0.05),(1, 0.05), (100, 0.01)]
result_lda = tune_rbf_pipeline(X_train, y_train, X_test, y_test, option_fine_lda, LinearDiscriminantAnalysis(n_components=9))
result_lda,table_lda = plot_rbf_bar_chart(result_lda, "LDA")
print(result_lda)
print(table_lda)


