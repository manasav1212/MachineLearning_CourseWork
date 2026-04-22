from lib import *
from torch.utils.data import DataLoader

all_tickers = ["WMT", "MSFT", "NVDA", "GOOGL"]

for ticker in all_tickers:
    X_train, X_test, y_train, y_test, metadata = fetch_data([ticker], "2021-01-01", "2025-12-31", 50)
    data_loader = DataLoader(JointDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(JointDataset(X_test, y_test), batch_size=32, shuffle=False)
    input_shape = X_train.shape[2]

    layers = [
        [('rnn', 64)],
        [('rnn', 128)],
        [('rnn', 64), ('linear', 64)],
        [('rnn', 64), ('rnn', 64)],
        [('rnn', 64), ('linear', 64), ('rnn', 64)],
        [('linear', 64), ('rnn', 64)]
    ]

    models = []
    for layer in layers:
        seed_everything(42)
        model = DynamicRNN(input_shape, layer, dropout=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
        model, loss = train_model(model, optimizer, data_loader, 200)
        models.append((model, loss, layer))

    for model, loss, layer in models:
        print(f"Results for layer config: {layer}")
        results = evaluate_model(model, test_loader, metadata)
        print_results(results)
        print("\n\n")
        
# Train and evaluate the best model
# It was observed that the best model was the one with a single RNN layer with 128 hidden units, so we will train that one again and plot the predictions and loss curve
for ticker in all_tickers:
    X_train, X_test, y_train, y_test, metadata = fetch_data([ticker], "2021-01-01", "2025-12-31", 50)
    data_loader = DataLoader(JointDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(JointDataset(X_test, y_test), batch_size=32, shuffle=False)
    input_shape = X_train.shape[2]
    layers = [('rnn', 128)]
    model = model = DynamicRNN(input_shape, layers, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    model, loss = train_model(model, optimizer, data_loader, 200)
    print(f"Results for Ticker: {ticker}")
    results = evaluate_model(model, test_loader, metadata)
    print_results(results)
    plot_predictions(model, test_loader, metadata)
    plot_loss(loss)
    print("\n================================\n")