
# list of imports
import pandas as pd
import numpy as np
import os 
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.processing import preprocess

class get_tfidf:
    def __new__(cls, train_set, test_set):
        self = super().__new__(cls)
        self.__init__(train_set, test_set)
        return self()

    def __init__(self, train_set, test_set):
        # tfidf vectorizer here
        self.train_set = train_set.copy()
        self.test_set = test_set.copy()

    def tfidf(self, train_set, test_set):
        if "answer" not in train_set.columns:
            train_set["answer"] = (
                train_set["FalseSent"] + " [SEP] " +
                train_set["OptionA"] + " [SEP] " +
                train_set["OptionB"] + " [SEP] " +
                train_set["OptionC"]
            )
        tfidf = TfidfVectorizer(binary=True, ngram_range=(1, 2), max_features=20000, min_df=2)
        x = tfidf.fit_transform(train_set["answer"]).toarray().astype(np.float32)
        return x

    def encodelabel(self, train_set):
        mapping = {"A": 0, "B": 1, "C": 2}

        # IMPORTANT: if we have "label", use it. "answer" is the concatenated text.
        if "label" in train_set.columns:
            labels = train_set["label"].astype(str).str.strip().values
        elif "answer" in train_set.columns:
            labels = train_set["answer"].astype(str).str.strip().values
        else:
            labels = np.array(train_set).astype(str)

        y = np.array([mapping[l] for l in labels], dtype=np.int64)
        return y    



    def __call__(self):
        seed = 40
        x = self.tfidf(self.train_set, self.test_set)
        y = self.encodelabel(self.train_set)
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.2, random_state=seed
        )
        return x_train, x_val, y_train, y_val


if __name__ == "__main__":

    x_train_raw = pd.read_csv('/Users/owenarink/Documents/University/neuralnets101/kaggle_commonsense/data/train_data.csv')

    # test_data_csv
    x_test_raw = pd.read_csv('/Users/owenarink/Documents/University/neuralnets101/kaggle_commonsense/data/test_data.csv') 
    
    # answer_csv
    y_train_raw = pd.read_csv('/Users/owenarink/Documents/University/neuralnets101/kaggle_commonsense/data/train_answers.csv')
    #print("answers columns:", y_train_raw.columns)
    #print("answers head:", y_train_raw.head())
    train_df, test_df = preprocess(x_train_raw, x_test_raw, y_train_raw)
    #print(train_df.shape)
    print('x_train_raw', x_train_raw)
    print('x_test_raw', x_test_raw)
    print('y_train_raw', y_train_raw)
    print('train_df', train_df)
    print('test_df', test_df)
    x_train, x_val, y_train, y_val = get_tfidf(train_df, test_df)
    print('x_train', x_train)
    print('x_val', x_val)
    print('y_train', y_train)
    print('y_val', y_val)
    print("x_train shape:", x_train.shape)
    

    print("x_train nonzeros:", np.count_nonzero(x_train), "/", x_train.size)

    row0 = x_train[0]
    nz = np.nonzero(row0)[0]
    print("row0 nonzero count:", len(nz))
    print("row0 indices:", nz[:20])
    print("row0 values:", row0[nz[:20]])

    
    print('tfidf completed\n')
    print('------------------------------------------------')
