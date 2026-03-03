
# list of imports
import pandas as pd
import numpy as np
import os 
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# sampel for processing

filename_train_data = open("/Users/owenarink/Documents/University/neuralnets101/kaggle_commonsense/data/train_data.csv")
train_data_csv = pd.read_csv(filename_train_data)
train_data_array = np.array(train_data_csv)
print(train_data_array.shape)

filename_test_data = open("/Users/owenarink/Documents/University/neuralnets101/kaggle_commonsense/data/test_data.csv")
test_data_csv = pd.read_csv(filename_test_data)
test_data_array = np.array(test_data_csv)
print(test_data_array.shape)

train_df = train_data_array
test_df = test_data_array


#concatenate train and test set first

df = pd.concat([train_df, test_df], ignore_index=True)



train_set = df[df.answer != 



tfidf = TfidfVectorizer(binary=True)
x = tfidf.fit_transform(train_set.answer).todense()
x_test = tfidf.transform(test_set.answer).todense()

