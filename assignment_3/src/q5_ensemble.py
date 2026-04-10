import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset
import torch
from lib import *

from torch.utils.data import DataLoader, random_split

df = pd.read_csv('../data/movie_data.csv', encoding='utf-8')

X_train, X_test, y_train, y_test = train_test_split(
    df['review'],
    df['sentiment'],
    test_size=0.3,
    random_state=7
)

tfidf = TfidfVectorizer(dtype=np.float32)

X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

dataset = JointDataset(X_train, y_train)

total = len(dataset)
data_loader = []

input_shape = X_train.shape[1]
seed_everything(42)

models = [  DynamicNeuralNet(input_shape, [64, 64, 64], [0.5, 0.5, 0.5]),
            DynamicNeuralNet(input_shape, [64, 64, 64], [0.4, 0.4, 0.4]),
            DynamicNeuralNet(input_shape, [64, 64, 64], [0.4, 0.5, 0.6]),
            DynamicNeuralNet(input_shape, [64, 64, 64], [0.3, 0.3, 0.3]),
            DynamicNeuralNet(input_shape, [64, 64, 64], [0.1, 0.2, 0.3]),
        ]
num_of_models = len(models)

idx = [total // num_of_models] * num_of_models
for i in range(total % num_of_models):
    idx[i] += 1
generator = torch.Generator().manual_seed(42)
split_dataset = random_split(dataset, idx, generator = generator)
data_loaders = [DataLoader(x, 64, shuffle=True) for x in split_dataset]
assert len(data_loaders) == num_of_models

start = time.perf_counter()
for i in range(num_of_models):
    print(f"Training model {i+1} with dropout rates {models[i].dropout_rates}")
    optimizer = torch.optim.Adam(models[i].parameters(), lr = 1e-4)
    models[i] = train_model(models[i], optimizer, data_loaders[i])
epoch_5_time= time.perf_counter() - start
epoch_5_accuracy = evaluate_ensembled_models(models, test_loader, "Bagging ensemble")

start = time.perf_counter()
for i in range(num_of_models):
    print(f"Training model {i+1} with dropout rates {models[i].dropout_rates}")
    optimizer = torch.optim.Adam(models[i].parameters(), lr = 1e-4)
    models[i] = train_model(models[i], optimizer, data_loaders[i], 20)
epoch_20_time= time.perf_counter() - start
epoch_20_accuracy = evaluate_ensembled_models(models, test_loader, "Bagging ensemble")


print(f"Total training time for 5 epochs = {epoch_5_time} sec")
print(f"Total training time for 20 epochs = {epoch_20_time} sec")
print(f"Accuracy for 5 epochs = {epoch_5_accuracy}")
print(f"Accuracy for 20 epochs = {epoch_20_accuracy}")

 
