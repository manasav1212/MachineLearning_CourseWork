from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from lib import *

df = pd.read_csv('../data/movie_data.csv', encoding='utf-8')

X_train, X_test, y_train, y_test = train_test_split(
    df['review'],
    df['sentiment'],
    test_size=0.3,
    random_state=7
)

def tokenizer(text):
    return text.split()

df = pd.read_csv('../data/movie_data.csv', encoding='utf-8')

X_train, X_test, y_train, y_test = train_test_split(
    df['review'],
    df['sentiment'],
    test_size=0.3,
    random_state=7
)

tfidf = TfidfVectorizer(dtype=np.float32, ngram_range= (1,1), stop_words= None, tokenizer= tokenizer)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)
lr = LogisticRegression(solver='liblinear', C = 10.0, penalty = 'l2' )
start = time.perf_counter()
lr.fit(X_train, y_train)
print("Time taken to train Logistic Regression: ", time.perf_counter() - start)
y_pred = lr.predict(X_test)
y_train_pred = lr.predict(X_train)
print("Logistic Regression Accuracy on Test Set: ", np.mean(y_pred == y_test))
print("Logistic Regression Accuracy on Train Set: ", np.mean(y_train_pred == y_train))