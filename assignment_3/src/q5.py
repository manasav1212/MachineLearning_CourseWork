import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset
import torch
from lib import *

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
train_loader = DataLoader(dataset, 64, shuffle=True,)
test_loader = DataLoader(JointDataset(X_test, y_test), batch_size=64, shuffle=False)

input_shape = X_train.shape[1]
layers = [32, 32, 32]
dropouts = [
    [0.2, 0.4, 0.5],
    [0.5, 0.5, 0.5],
    [0.4, 0.4, 0.4],
    [0.1, 0.3, 0.6],
    [0.3, 0.3, 0.3],
    [0.3, 0.4, 0.5],
    [0.4, 0.5, 0.6],
    [0.6, 0.6, 0.6],
    [0.5, 0.5, 0.6],
    [0.5, 0.6, 0.7],
    [0.7, 0.7, 0.7],
]
droupout_results = []
test_accuracy = []
train_accuracy = []
for dropout_rate in dropouts:
    print(f"===============Training model for {dropout_rate} =======================")
    seed_everything(42)
    model = DynamicNeuralNet(input_shape, layers, dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    train_model(model, optimizer, train_loader)

    # Evaluate the model
    train_accuracy.append(evaluate_model(model, train_loader, "Training"))
    test_accuracy.append(evaluate_model(model, test_loader, "Test"))
    droupout_results.append(dropout_rate)
result = pd.DataFrame({"dropout_rate": droupout_results, "train_accuracy": train_accuracy, "test_accuracy": test_accuracy})
print(result)
 
