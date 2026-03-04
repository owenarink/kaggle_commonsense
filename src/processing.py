import numpy as np
import csv 
import pandas as pd


# sampel for processing
def preprocess(train_data_csv: pd.DataFrame,  answers_csv: pd.DataFrame, test_data_csv: pd.DataFrame):
    # do some preprocessing steps here

    train = train_data_csv.copy()
    test = test_data_csv.copy()
    ans = answers_csv.copy()

    train.columns = ["id", "statement", "A", "B", "C"]
    test.columns  = ["id", "statement", "A", "B", "C"]
    ans.columns = ["id", "label"]

    train_df = train.merge(ans, on="id", how="inner")
    test_df = test
    return train_df, test_df

x_train_raw = pd.read_csv('/Users/owenarink/Documents/University/neuralnets101/kaggle_commonsense/data/train_data.csv')
x_test_raw = pd.read_csv('/Users/owenarink/Documents/University/neuralnets101/kaggle_commonsense/data/test_data.csv') 
y_train_raw = pd.read_csv('/Users/owenarink/Documents/University/neuralnets101/kaggle_commonsense/data/train_answers.csv')


train_df, test_df = preprocess(x_train_raw, y_train_raw, x_test_raw)
print(train_df.shape)
print(test_df.shape)
print(train_df)
print(test_df)
#print('x_train_preprocessed: ', train_df, 'y_train_preprocessed: ', test_df)


    #df = pd.concat([train_df, test_df], ignore_index=True)



    #train_set = df[df.answer != ]
    #test_set = df[df.answer == ]




