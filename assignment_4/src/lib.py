
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import torch
import yfinance as yf
import random
from torch.utils.data import Dataset
from torch import nn

import matplotlib.pyplot as plt
import numpy as np
import time

def split_and_scale(df, window_size = 50):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    x_cols = ['Open', 'High', 'Low', 'Volume', 'Close']
    y_col = ['Close']
    
    X_raw = df[x_cols]
    y_raw = df[y_col]
    
    X_scaled = x_scaler.fit_transform(X_raw)
    y_scaled = y_scaler.fit_transform(y_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)
    
    y_train = y_train[window_size:]
    
    X_test = np.vstack((X_train[-window_size:], X_test))
    
    return X_train, X_test, y_train, y_test, (x_scaler, y_scaler)

def create_rolling_window(X, window_size):
    X_windowed = []
    for i in range(len(X) - window_size):
        X_windowed.append(X[i:i+window_size])
    return np.array(X_windowed)

def preprocess_data(raw_data : dict, window_size : int, x_cols : list = ['Open', 'High', 'Low', 'Volume', 'Close']):
    ticker_count = len(raw_data)
    X_train = np.empty((0, window_size, len(x_cols)))
    X_test = np.empty_like(X_train)
    y_train = np.empty((0, 1))
    y_test = np.empty_like(y_train)

    metadata = {}

    for i, (ticker, df) in enumerate(raw_data.items()):
        print(f'{i}. {ticker}: {len(df)} rows')
        X_train_i, X_test_i, y_train_i, y_test_i, scalers = split_and_scale(df, window_size)
        X_train_i = create_rolling_window(X_train_i, window_size)
        X_test_i = create_rolling_window(X_test_i, window_size)

        # Uncomment this if we want to train single model for all companies using one-hot-encoding
        # one_hot = np.zeros((len(X_train_i), window_size, ticker_count))
        # one_hot[:, :, i] = 1
        # X_train_i = np.concatenate([X_train_i, one_hot], axis=2)

        # one_hot = np.zeros((len(X_test_i), window_size, ticker_count))
        # one_hot[:, :, i] = 1
        # X_test_i = np.concatenate([X_test_i, one_hot], axis=2)

        X_train = np.concatenate([X_train, X_train_i], axis=0)
        X_test = np.concatenate([X_test, X_test_i], axis=0)
        y_train = np.concatenate([y_train, y_train_i], axis=0)
        y_test = np.concatenate([y_test, y_test_i], axis=0)

        train_start = len(X_train) - len(X_train_i)
        train_end = train_start + len(X_train_i)
        test_start = len(X_test) - len(X_test_i)
        test_end = test_start + len(X_test_i)
        metadata[ticker] = { 
            'scalers': scalers,
            'index': i,
            # This will be useful if we use one-hot-encoding to know which row belongs to which company
            'train_range': (train_start, train_end),
            'test_range': (test_start, test_end)
        }
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32), metadata

def fetch_data(tickers: list[str], start_date, end_date, window_size = 50):
    data = {name: yf.download(name, start=start_date, end=end_date) for name in tickers}
    return preprocess_data(data, window_size)

def seed_everything(seed=42):
    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class JointDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __str__(self):
        result = ''
        for x,y in zip(self.x, self.y):
             result += 'x: ' + str(x) + '\t' + 'y: ' + str(y) + '\n'
        return result

def evaluate_model(model, test_loader, metadata, range_key='test_range', device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Metrics:
      - RMSE: Root Mean Squared Error (dollar units)
      - MAE:  Mean Absolute Error (dollar units)
      - MAPE: Mean Absolute Percentage Error (%)
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy().reshape(-1, 1))

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    results = {}

    for ticker, meta in metadata.items():
        # We added test_range during processing to know which row belongs to which company.
        start, end = meta[range_key]
        # Need the scaler to reverse the min-max normalization
        y_scaler = meta['scalers'][1]

        preds_scaled = all_preds[start:end]
        targets_scaled = all_targets[start:end]

        # Inverse scaler to get the dollar amount
        preds = y_scaler.inverse_transform(preds_scaled).flatten()
        targets = y_scaler.inverse_transform(targets_scaled).flatten()

        errors = preds - targets
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / targets)) * 100
        actual_changes = np.diff(targets)
        pred_changes = np.diff(preds)
        dir_acc = np.mean(np.sign(actual_changes) == np.sign(pred_changes)) * 100

        results[ticker] = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'DirAcc': dir_acc,
        }

    # Overall metrics; Might not be meaningful since different companies have different price ranges except MAPE. 
    # Useful if used one-hot-encoding and training single model for all companies together to see overall performance across all companies.
    results['OVERALL'] = {
        metric: np.mean([results[t][metric] for t in metadata])
        for metric in ['RMSE', 'MAE', 'MAPE', 'DirAcc']
    }

    return results


def print_results(results):
    print(f"{'Ticker':<10} {'RMSE':>10} {'MAE':>10} {'MAPE(%)':>10} {'DirAcc(%)':>12}")
    print("-" * 56)
    for ticker, m in results.items():
        print(f"{ticker:<10} {m['RMSE']:>10.2f} {m['MAE']:>10.2f} "
              f"{m['MAPE']:>10.2f} {m['DirAcc']:>12.2f}")

def plot_predictions(model, test_loader, metadata, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy().reshape(-1, 1))
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # NUmber of companies
    n = len(metadata)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    _, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
    
    # Subplot needs array
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for ax, (ticker, meta) in zip(axes, metadata.items()):
        start, end = meta['test_range']
        y_scaler = meta['scalers'][1]
        
        preds_scaled = all_preds[start:end]
        targets_scaled = all_targets[start:end]
        
        preds = y_scaler.inverse_transform(preds_scaled).flatten()
        targets = y_scaler.inverse_transform(targets_scaled).flatten()
        
        days = np.arange(len(targets))
        ax.plot(days, targets, label='Actual', color='tab:blue', linewidth=1.5)
        ax.plot(days, preds, label='Predicted', color='tab:orange', linewidth=1.5, alpha=0.8)
        
        ax.set_title(f'{ticker}')
        ax.set_xlabel('Test day')
        ax.set_ylabel('Close price ($)')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

class TunedRnn(torch.nn.Module):

    def __init__(self, input_size, dropout_rate):
        super().__init__()
        self.r1 = nn.RNN(input_size, 32, 1, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(32, 1)

    def forward(self, X):
        rnn_out, hidden = self.r1(X)
        output = self.dropout(hidden[-1])
        return self.output(output)

def plot_loss(loss, figure_size=(9, 5)):
    plt.figure(figsize= figure_size)
    plt.plot(loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss(Log scale)')
    plt.yscale('log')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

def train_model(model, optimizer, data_loader, epochs = 200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f'Device found: {device}')
    model = model.to(device)
    L = torch.nn.MSELoss()
    epoch_losses = []
    model.train()
    start = time.perf_counter()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            y_batch = y_batch.float().to(device)
            output = model(X_batch)
            loss = L(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_losses.append(epoch_loss / n_batches)
    print(f'Train time = {time.perf_counter() - start} sec')
    return model, epoch_losses

class DynamicRNN(nn.Module):

    def __init__(self, input_size, layers, dropout=0.2):
        super().__init__()
        
        if not any(layer_type == 'rnn' for layer_type, _ in layers):
            raise ValueError("At least one layer must be RNN")
        
        self.layer_types = [layer_type for layer_type, _ in layers]
        self.layers = nn.ModuleList()
        
        # If there is a linear after the last RNN, we will take the output of the last RNN as input to that linear layer
        self.last_rnn_idx = max(
            i for i, (layer_type, _) in enumerate(layers) if layer_type == 'rnn'
        )
        
        # Current dim starts as input_size and gets updated after each layer
        current_dim = input_size
        for layer_type, output_dim in layers:
            if layer_type == 'linear':
                self.layers.append(nn.Linear(current_dim, output_dim))
            elif layer_type == 'rnn':
                self.layers.append(nn.RNN(current_dim, output_dim, num_layers=1, batch_first=True))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            current_dim = output_dim
        
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(current_dim, 1)
    
    def forward(self, X):
        for i, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            if layer_type == 'linear':
                X = torch.relu(layer(X))
            else:
                rnn_out, hidden = layer(X)
                if i == self.last_rnn_idx:
                    X = hidden[-1]
                else:
                    X = rnn_out
            X = self.dropout(X)
        
        return self.output(X)

class DynamicGRU(nn.Module):

    def __init__(self, input_size, layers, dropout=0.2):
        super().__init__()
        
        if not any(layer_type == 'gru' for layer_type, _ in layers):
            raise ValueError("At least one layer must be GRU")
        
        self.layer_types = [layer_type for layer_type, _ in layers]
        self.layers = nn.ModuleList()
        
        # If there is a linear after the last RNN, we will take the output of the last RNN as input to that linear layer
        self.last_gru_idx = max(
            i for i, (layer_type, _) in enumerate(layers) if layer_type == 'gru'
        )
        
        # Current dim starts as input_size and gets updated after each layer
        current_dim = input_size
        for layer_type, output_dim in layers:
            if layer_type == 'linear':
                self.layers.append(nn.Linear(current_dim, output_dim))
            elif layer_type == 'gru':
                self.layers.append(nn.GRU(current_dim, output_dim, num_layers=1, batch_first=True))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            current_dim = output_dim
        
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(current_dim, 1)
    
    def forward(self, X):
        for i, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            if layer_type == 'linear':
                X = torch.relu(layer(X))
            else:
                gru_out, hidden = layer(X)
                if i == self.last_gru_idx:
                    X = hidden[-1]
                else:
                    X = gru_out
            X = self.dropout(X)
        
        return self.output(X)

class DynamicLSTM(nn.Module):
    def __init__(self, input_size, layers, dropout=0.2):
        super().__init__()
        
        if not any(layer_type == 'lstm' for layer_type, _ in layers):
            raise ValueError("At least one LSTM layer required")
        
        self.layer_types = [layer_type for layer_type, _ in layers]
        self.layers = nn.ModuleList()
        
        self.last_lstm_idx = max(
            i for i, (layer_type, _) in enumerate(layers) if layer_type == 'lstm'
        )
        
        current_dim = input_size
        for layer_type, output_dim in layers:
            if layer_type == 'linear':
                self.layers.append(nn.Linear(current_dim, output_dim))
            elif layer_type == 'lstm':
                self.layers.append(nn.LSTM(current_dim, output_dim, num_layers=1, batch_first=True))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            current_dim = output_dim
        
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(current_dim, 1)
    
    def forward(self, X):
        for i, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            if layer_type == 'linear':
                X = torch.relu(layer(X))
            else:
                lstm_out, (hidden, cell) = layer(X)
                if i == self.last_lstm_idx:
                    X = hidden[-1]
                else:
                    X = lstm_out
            X = self.dropout(X)
        
        return self.output(X)