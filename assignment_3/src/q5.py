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
data_loader = DataLoader(dataset, 64, shuffle=True,)
test_loader = DataLoader(JointDataset(X_test, y_test), batch_size=64, shuffle=False)

input_shape = X_train.shape[1]

# Trying out different dropouts with the baseline FNN model
models = {'a': DynamicNeuralNet(input_shape, [64,64], [0.1, 0.5]),
          'b': DynamicNeuralNet(input_shape, [64,64], [0.2, 0.5]),
          'c': DynamicNeuralNet(input_shape, [64,64], [0.3, 0.7]),
          'd': DynamicNeuralNet(input_shape, [64,64], [0.4, 0.7]),
          'e': DynamicNeuralNet(input_shape, [64,64], [0.4, 0.4]),
          'f': DynamicNeuralNet(input_shape, [64,64], [0.5, 0.5]),
          }
for name, model in models.items():
    print(f"===============Training model {name} =======================")
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    train_model(model, optimizer, data_loader)

    # Evaluate the model

    test_loader = DataLoader(JointDataset(X_test, y_test), batch_size=64, shuffle=False)
    evaluate_model(model, data_loader, "Training")
    evaluate_model(model, test_loader)

# Using bagging to train multiple models and ensemble them together
 
