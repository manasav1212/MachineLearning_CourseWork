import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch

tickers = ["AMZN", "MSFT", "NVDA", "GOOGL"]
data = yf.download(tickers, start="2025-01-01", end="2026-01-01")
closing_data = data['Close']
closing_data.dropna()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(closing_data)

X_train, X_test = train_test_split(scaled_data, test_size=0.2, random_state=7)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

print(X_train_tensor.shape)
print(X_test_tensor.shape)

torch.save(X_train_tensor, "X_train.pt")
torch.save(X_test_tensor, "X_test.pt")