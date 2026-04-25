#  Data prepration and model training code for question 1
#  This has been implemented using 4 functions in lib.py
#  1. split_and_scale
#  2. create_rolling_window
#  3. preprocess_data
#  4. fetch_data

from lib import *

all_tickers = ["WMT", "MSFT", "NVDA", "GOOGL"]

for ticker in all_tickers:
    X_train, X_test, y_train, y_test, metadata = fetch_data([ticker], "2021-01-01", "2025-12-31", 50)
    print(f"Ticker: {ticker}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Metadata: {metadata}")
    print(type(X_train), type(y_train), type(X_test), type(y_test))
    print("\n================================\n")