import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import time
from sklearn.feature_extraction.text import TfidfVectorizer

seed = 42
torch.manual_seed(seed)

class NeuralNet(torch.nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 32)
        self.output = torch.nn.Linear(32, 2)
    
    def forward(self, X):
        X = self.fc1(X)
        X = torch.nn.functional.relu(X)
        X = torch.nn.functional.relu(self.fc2(X))
        X = torch.nn.functional.relu(self.fc3(X))
        return self.output(X)

class reviewDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

k_folds = 5
num_epochs = 1
loss_function = nn.CrossEntropyLoss()
kfold = KFold(n_splits=k_folds, shuffle=True)
test_accuracy = []
train_time = []
train_accuracy = []

df = pd.read_csv('clean_movie_data.csv', encoding='utf-8')
X_text = df['clean_text']
y = df['sentiment']

def run_kfold():
    for fold, (train_ids, test_ids) in enumerate(kfold.split(df)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        start_time = time.time()
        train_text = [X_text[i] for i in train_ids]
        test_text = [X_text[i] for i in test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]
        
        tfidf = TfidfVectorizer(max_features=15000)
        X_train = tfidf.fit_transform(train_text).toarray()
        X_test = tfidf.transform(test_text).toarray()
        
        train_dataset = reviewDataset(X_train, y_train)
        test_dataset = reviewDataset(X_test, y_test)
        
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=10)
        
        input_shape = X_train.shape[1]
        network = NeuralNet(input_shape)
        #layer = [32,32,32]
        #network = DynamicNeuralNet(input_shape, layer)
        
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4, weight_decay=0.00001)
        
        for _ in range(0, num_epochs):
            current_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                optimizer.zero_grad()
                outputs = network(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
                if i % 500 == 499:
                    current_loss = 0.0
        
        end_time = time.time()
        fold_time = end_time-start_time
        train_time.append(fold_time)
        print(f"Time taken by fold {fold}: {fold_time:.2f} seconds")
        
        network.eval()
        train_correct, train_total = 0,0
        with torch.no_grad():
            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                outputs = network(inputs)
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
        print('Train Accuracy for fold %d: %d %%' % (fold, 100.0 * train_correct / train_total))
        train_accuracy.append(100.0 * train_correct / train_total)
        
        save_path = f'./model-fold-{fold}.pth'
        torch.save(network.state_dict(), save_path)
        correct, total = 0, 0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, targets = data
                outputs = network(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print('Test Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        test_accuracy.append(100.0 * (correct / total))

run_kfold()   
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
total_time = sum(train_time)
print(f"Total training time : {total_time:.2f} seconds")
avg_train_accu = sum(train_accuracy)/len(train_accuracy)
print(f'Average Train accuracy: {avg_train_accu} %')
avg_test_accu = sum(test_accuracy)/len(test_accuracy)
print(f'Average Test accuracy: {avg_test_accu} %')