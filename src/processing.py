import numpy as np
import csv 
import pandas as pd


filename_train_data = open("/Users/owenarink/Documents/University/neuralnets101/kaggle_commonsense/data/train_data.csv")
train_data_csv = pd.read_csv(filename_train_data)
train_data_array = np.array(train_data_csv)
print(train_data_array.shape)
# 8000 x 5 (ROWSxCOLUMNS)
# id, FalseSent, OptionA, OptionB, OptionC

train_data_array = train_data_array[:, 1:]
print(train_data_csv_array.shape)
filename_train_answers = open("/Users/owenarink/Documents/University/neuralnets101/kaggle_commonsense/data/train_answers.csv")
train_answers_csv = pd.read_csv(filename_train_answers)
train_answers_array = np.array(train_answers_csv)
train_answers_array = train_answers_array[:, 1:]
print(train_answers_array.shape)




