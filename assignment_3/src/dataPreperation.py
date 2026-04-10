import numpy as np
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

def preprocessor(text):
     text = re.sub('<[^>]*>', '', text)
     emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                            text)
     text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
     return text

directoryPath = os.path.dirname(os.path.abspath(__file__))
basepath = "..\\data\\aclImdb"
labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()
count = CountVectorizer()
rows = []

for s in ('test', 'train'):
     for l in ('pos', 'neg'):
        path = os.path.join(directoryPath, basepath, s, l)
        for file in sorted(os.listdir(path)):
             with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                 txt = infile.read()
             rows.append([txt, labels[l]])
df = pd.DataFrame(rows, columns=['review', 'sentiment'])

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

df = pd.read_csv('movie_data.csv', encoding='utf-8')

df['clean_text'] = df['review'].apply(preprocessor)
df.to_csv('clean_movie_data.csv', index=False, encoding='utf-8')

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
np.set_printoptions(precision=2)
transformed_text = tfidf.fit_transform(count.fit_transform(df['clean_text']))

X_train, X_test, y_train, y_test = train_test_split(transformed_text, df['sentiment'], test_size = 0.3, random_state = 42)
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)
