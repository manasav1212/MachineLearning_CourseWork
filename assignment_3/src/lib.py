import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class JointDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y.values

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx].toarray().squeeze(), self.y[idx].squeeze()
    
    def __str__(self):
        result = ''
        for x,y in zip(self.x, self.y):
             'x: ' + str(x) + '\t' + 'y: ' + str(y) + '\n'
        return result

class JointSparseDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y.values

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __str__(self):
        result = ''
        for x,y in zip(self.x, self.y):
             'x: ' + str(x) + '\t' + 'y: ' + str(y) + '\n'
        return result
    
import scipy.sparse
import time

def sparse_collate(batch):
    
    x_sparse = scipy.sparse.vstack([item[0] for item in batch])
    
    x_dense = torch.tensor(x_sparse.toarray(), dtype=torch.float32)
    
    y_batch = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    
    return x_dense, y_batch

class NeuralNet(torch.nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        # self.fc3 = torch.nn.Linear(64, 64)
        # self.dropout = torch.nn.Dropout(0.5)
        self.output = torch.nn.Linear(64, 1)
    
    def forward(self, X):
        X = self.fc1(X)
        X = torch.nn.functional.relu(X)
        # X = self.dropout(X)
        X = torch.nn.functional.relu(self.fc2(X))
        # X = self.dropout(X)
        # X = torch.nn.functional.relu(self.fc3(X))
        # X = self.dropout(X)
        # X = torch.nn.functional.relu(self.fc4(X))
        # X = self.dropout(X)
        return self.output(X)
    
class NeuralNetWithDropout(torch.nn.Module):

    def __init__(self, input_size, dropout_rate):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.dropout1 = torch.nn.Dropout(dropout_rate[0])
        self.dropout2 = torch.nn.Dropout(dropout_rate[1])
        self.dropout3 = torch.nn.Dropout(dropout_rate[2])
        self.output = torch.nn.Linear(64, 1)

    def forward(self, X):
        X = self.fc1(X)
        X = torch.nn.functional.relu(X)
        X = self.dropout1(X)
        X = torch.nn.functional.relu(self.fc2(X))
        X = self.dropout2(X)
        X = torch.nn.functional.relu(self.fc3(X))
        X = self.dropout3(X)
        return self.output(X)

class DynamicNeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rates=None):
        super().__init__()
        self.layers = torch.nn.ModuleList() 
        current_dim = input_size
        self.use_dropout = dropout_rates is not None
        if (self.use_dropout):
            assert len(hidden_layers) == len(dropout_rates), "Length mismatch"
            self.dropouts = torch.nn.ModuleList()

        for i in range(len(hidden_layers)):
            self.layers.append(torch.nn.Linear(current_dim, hidden_layers[i]))
            
            if self.use_dropout:
                self.dropouts.append(torch.nn.Dropout(dropout_rates[i]))
            
            current_dim = hidden_layers[i]

        self.output = torch.nn.Linear(current_dim, 1)
        
    def forward(self, X):
        if self.use_dropout:
            for layer, dropout in zip(self.layers, self.dropouts):
                X = layer(X)
                X = torch.nn.functional.relu(X)
                X = dropout(X)
        else:
            for layer in self.layers:
                X = layer(X)
                X = torch.nn.functional.relu(X)
            
        return self.output(X)
    
def train_model(model, optimizer, data_loader, epochs = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f'Device found: {device}')
    model = model.to(device)
    L = torch.nn.BCEWithLogitsLoss()

    model.train()
    start = time.perf_counter()
    for epoch in range(epochs):
        for (X_batch, y_batch) in data_loader:
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            y_batch = y_batch.unsqueeze(1).float().to(device)
            output = model(X_batch)
            loss = L(output, y_batch)
            loss.backward()
            optimizer.step()

    print(f'Train time = {time.perf_counter() - start} sec')
    return model

def evaluate_model(model, data_loader, title = 'Test'):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0

    with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                y_batch = y_batch.unsqueeze(1).float().to(device)

                probability = torch.sigmoid(output)
                y_pred =  (probability >= 0.5)

                match_count = (y_batch == y_pred).sum().item()
                correct += match_count
                total += X_batch.shape[0]

    accuracy = correct / total
    print(f"{title} Accuracy:", accuracy)