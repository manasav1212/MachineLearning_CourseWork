import yfinance as yf

tickers = ["AMZN", "MSFT", "NVDA", "GOOGL"]
data = yf.download(tickers, start="2025-01-01", end="2026-01-01")
closing_data = data['Close']