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
#  Baseline model tuning
decays = [1e-5,1e-4,5e-4,5e-5, 1e-6]
layers = [[32,32,32],[64,64,64],[32,32,32,32]]
accuracy = []
train_accuracy = []
layer_results = []
decay_results = []
for layer in layers:
    for decay in decays:
        print(f"===============Training model with {decay} =======================")
        seed_everything(42) 
        model = DynamicNeuralNet(input_shape, layer)
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay= decay)
        train_model(model, optimizer, train_loader)

        # Evaluate the model
        train_accuracy.append(evaluate_model(model, train_loader, "Training"))
        accuracy.append(evaluate_model(model, test_loader, "Test"))
        layer_results.append(layer)
        decay_results.append(decay)
res = pd.DataFrame({"hidden_layers": layer_results, "decay": decay_results, "accuracy": accuracy, "train_accuracy": train_accuracy})
print(res)

# Figurin out the number of epochs for training
seed_everything(42)
baseline_model = DynamicNeuralNet(input_shape, [32, 32, 32] )
optimizer = torch.optim.Adam(baseline_model.parameters(), lr = 1e-4, weight_decay= 1e-5)
_, history = train_model2(baseline_model, optimizer, train_loader, 20, test_loader)

# PLotting the training and validation loss curves
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
    
# Plot training loss
plt.plot(history['train_loss'], label='Training Loss', color='blue', marker='o')

# Plot validation loss if it exists
if len(history['val_loss']) > 0:
    plt.plot(history['val_loss'], label='Validation Loss', color='orange', marker='s')
    
plt.title("Baseline Model Loss Curves")
plt.xlabel('Epoch')
plt.ylabel('Loss (BCEWithLogits)')

# Ensure x-axis only shows integer epoch numbers
plt.xticks(range(len(history['train_loss']))) 

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
