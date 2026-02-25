import numpy as np
import csv 
import pandas as pd


f = open("/Users/owenarink/Documents/University/neuralnets101/kaggle_commonsense/data/train_data.csv")
train_data_csv = pd.read_csv(f)
train_data_csv_array = np.asarray(train_data_csv)
print(train_data_csv.shape())
