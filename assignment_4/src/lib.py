
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf

def split_and_scale(df, window_size = 50):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_cols = ['Open', 'High', 'Low', 'Volume', 'Close']
    y_col = ['Close']
    X = df[x_cols]
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=False)
    y_train = y_train.iloc[window_size:]
    X_test = pd.concat([X_train.iloc[-window_size:], X_test])
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)
    return X_train, X_test, y_train, y_test, (x_scaler, y_scaler)

def create_rolling_window(X, window_size):
    X_windowed = []
    for i in range(len(X) - window_size):
        X_windowed.append(X[i:i+window_size])
    return np.array(X_windowed)

def preprocess_data(raw_data : dict, window_size : int, x_cols : list = ['Open', 'High', 'Low', 'Volume', 'Close']):
    ticker_count = len(raw_data)
    X_train = np.empty((0, window_size, len(x_cols) + len(raw_data)))
    X_test = np.empty_like(X_train)
    y_train = np.empty((0, 1))
    y_test = np.empty_like(y_train)

    metadata = {}

    for i, (ticker, df) in enumerate(raw_data.items()):
        print(f'{i}. {ticker}: {len(df)} rows')
        X_train_i, X_test_i, y_train_i, y_test_i, scalers = split_and_scale(df, window_size)
        X_train_i = create_rolling_window(X_train_i, window_size)
        X_test_i = create_rolling_window(X_test_i, window_size)

        one_hot = np.zeros((len(X_train_i), window_size, ticker_count))
        one_hot[:, :, i] = 1
        X_train_i = np.concatenate([X_train_i, one_hot], axis=2)

        one_hot = np.zeros((len(X_test_i), window_size, ticker_count))
        one_hot[:, :, i] = 1
        X_test_i = np.concatenate([X_test_i, one_hot], axis=2)

        X_train = np.concatenate([X_train, X_train_i], axis=0)
        X_test = np.concatenate([X_test, X_test_i], axis=0)
        y_train = np.concatenate([y_train, y_train_i], axis=0)
        y_test = np.concatenate([y_test, y_test_i], axis=0)


        test_start = len(X_test) - len(X_test_i)
        test_end = test_start + len(X_test_i)
        metadata[ticker] = { 
            'scalers': scalers,
            'index': i,
            'test_range': (test_start, test_end)
        }
    return X_train, X_test, y_train, y_test, metadata

def fetch_data(tickers: list[str], start_date, end_date, window_size = 50):
    data = {name: yf.download(name, start=start_date, end=end_date) for name in tickers}
    return preprocess_data(data, window_size)
