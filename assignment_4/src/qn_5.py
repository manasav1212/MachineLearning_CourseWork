from lib import *
from torch.utils.data import DataLoader

tickers = ["WMT", "MSFT", "NVDA", "GOOGL"]

for ticker in tickers:
    print(f"Training and evaluating for {ticker}")
    X_train, X_test, y_train, y_test, metadata = fetch_data([ticker], "2021-01-01", "2025-12-31", 50)
    data_loader = DataLoader(JointDataset(X_train, y_train), batch_size=32, shuffle=True)
    train_eval_loader = DataLoader(JointDataset(X_train, y_train), batch_size=32, shuffle=False)
    test_loader = DataLoader(JointDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    layers = [
    [("lstm", 64), ('linear', 128)],
    [("lstm", 64), ('linear', 64)],
    [('lstm', 64)],
    [('lstm', 64), ('lstm', 64)],
    [('lstm', 64), ('lstm', 64), ('linear', 64)],
    ]
    
    models = []
    
    for layer in layers:
        seed_everything(42)
        model = DynamicLSTM(input_size=X_train.shape[2], layers=layer, dropout=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model, loss = train_model(model, optimizer, data_loader, 200)
        models.append((model, loss, layer))

    for model, loss, layer in models:
        print(f"Test Results for layer config: {layer}")
        results = evaluate_model(model, test_loader, metadata)
        print_results(results)
        print("\n\n")
        print(f"\n\nTraining Error for Ticker: {ticker}")
        results = evaluate_model(model, train_eval_loader, metadata, range_key='train_range')
        print_results(results)
        plot_predictions(model, test_loader, metadata)


layer_1 = [("lstm", 64), ('linear', 128)]
tickers_1 = ['NVDA', 'GOOGL']
for ticker in tickers_1:
    X_train, X_test, y_train, y_test, metadata = fetch_data([ticker], "2021-01-01", "2025-12-31", 50)
    data_loader = DataLoader(JointDataset(X_train, y_train), batch_size=32, shuffle=True)
    train_eval_loader = DataLoader(JointDataset(X_train, y_train), batch_size=32, shuffle=False)
    test_loader = DataLoader(JointDataset(X_test, y_test), batch_size=32, shuffle=False)
    input_shape = X_train.shape[2]
    layers = layer_1
    seed_everything(42)
    model = model = DynamicLSTM(input_shape, layers, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    model, loss = train_model(model, optimizer, data_loader, 200)
    print(f"Test Error for Ticker: {ticker}")
    results = evaluate_model(model, test_loader, metadata)
    print_results(results)
    print(f"\n\nTraining Error for Ticker: {ticker}")
    results = evaluate_model(model, train_eval_loader, metadata, range_key='train_range')
    print_results(results)
    plot_predictions(model, test_loader, metadata)
    plot_loss(loss)
    print("\n================================\n")

layer_2 = [('lstm', 64)]
tickers_2 = ['MSFT', 'WMT']
for ticker in tickers_2:
    X_train, X_test, y_train, y_test, metadata = fetch_data([ticker], "2021-01-01", "2025-12-31", 50)
    data_loader = DataLoader(JointDataset(X_train, y_train), batch_size=32, shuffle=True)
    train_eval_loader = DataLoader(JointDataset(X_train, y_train), batch_size=32, shuffle=False)
    test_loader = DataLoader(JointDataset(X_test, y_test), batch_size=32, shuffle=False)
    input_shape = X_train.shape[2]
    layers = layer_2
    seed_everything(42)
    model = model = DynamicLSTM(input_shape, layers, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    model, loss = train_model(model, optimizer, data_loader, 200)
    print(f"Test Error for Ticker: {ticker}")
    results = evaluate_model(model, test_loader, metadata)
    print_results(results)
    print(f"\n\nTraining Error for Ticker: {ticker}")
    results = evaluate_model(model, train_eval_loader, metadata, range_key='train_range')
    print_results(results)
    plot_predictions(model, test_loader, metadata)
    plot_loss(loss)
    print("\n================================\n")