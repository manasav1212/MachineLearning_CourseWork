import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from lib import *
from sklearn.model_selection import KFold
import time
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv(f'{DATA_FILE_PATH}/movie_data.csv', encoding='utf-8')
X = df['clean_text']
y = df['sentiment']

def run_kfold(k_fold, epochs, X, y):
    kfold = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    fold_times = []
    training_accuracy= []
    test_accuracy = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(X)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        start_time = time.perf_counter()
        fold_X_train = X.iloc[train_ids]
        fold_X_val = X.iloc[val_ids]
        fold_y_train = y.iloc[train_ids]
        fold_y_val = y.iloc[val_ids]

        tfidf = TfidfVectorizer(dtype=np.float32)
        fold_X_train = tfidf.fit_transform(fold_X_train)
        fold_X_val = tfidf.transform(fold_X_val)

        # generator for reproducibility
        generator = torch.Generator().manual_seed(42)
        fold_train_loader = DataLoader(JointDataset(fold_X_train, fold_y_train), batch_size=64, shuffle=True, generator=generator)
        fold_val_loader = DataLoader(JointDataset(fold_X_val, fold_y_val), batch_size=64, shuffle=False)

        seed_everything(42)
        input_size = fold_X_train.shape[1]
        model = DynamicNeuralNet(input_size, [32, 32, 32])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

        model = train_model(model, optimizer, fold_train_loader, epochs=epochs)
        fold_time = time.perf_counter() - start_time
        fold_times.append(fold_time)

        # Now measure the validation and training accuracy for this fold
        train_acc = evaluate_model(model, fold_train_loader)
        val_acc = evaluate_model(model, fold_val_loader)
        training_accuracy.append(train_acc)
        test_accuracy.append(val_acc)

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_fold} FOLDS')
    print('-======================================================')
    print(f"Total CV training time: {sum(fold_times)} seconds")
    print(f"Average Train Accuracy: {(np.mean(training_accuracy) * 100)}%")
    print(f"Average Validation Accuracy: {(np.mean(test_accuracy) * 100)}%")


run_kfold(k_fold=5, epochs=5, X=X, y=y)
run_kfold(k_fold=6, epochs=5, X=X, y=y)
run_kfold(k_fold=7, epochs=5, X=X, y=y)
