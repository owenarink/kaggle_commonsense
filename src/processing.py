import numpy as np
import csv 
import pandas as pd


# sampel for processing
def preprocess(train_data_csv: pd.DataFrame, test_data_csv: pd.DataFrame, answers_csv: pd.DataFrame): # do some preprocessing steps here
    train = train_data_csv.copy()
    test = test_data_csv.copy()
    ans = answers_csv.copy()
    
    required_cols = ["id", "FalseSent", "OptionA", "OptionB", "OptionC"]
    for df_name, df in [("train", train), ("test", test)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{df_name}_data_csv is missing columns: {missing}. Found: {list(df.columns)}")

    # Answers expected: make sure id and answer in the ans columns
    if "id" not in ans.columns or "answer" not in ans.columns:
        raise ValueError(f"answers_csv must have columns ['id','answer']. Found: {list(ans.columns)}")

    # clean basic text fields
    for col in ["FalseSent", "OptionA", "OptionB", "OptionC"]:
        train[col] = train[col].astype(str).str.strip()
        test[col]  = test[col].astype(str).str.strip()

    ans["answer"] = ans["answer"].astype(str).str.strip()

    # merge labels to train
    train_df = train.merge(ans[["id", "answer"]], on="id", how="inner")

    # Rename label column to something consistent
    train_df = train_df.rename(columns={"answer": "label"})

    # Optional: sanity check labels
    bad = train_df[~train_df["label"].isin(["A", "B", "C"])]
    if len(bad) > 0:
        raise ValueError(f"Found invalid labels (not A/B/C). Example rows:\n{bad.head()}")

    test_df = test[required_cols].copy()
    train_df = train_df[required_cols + ["label"]].copy()

    return train_df, test_df


if __name__ == "__main__":
    # train_data_csv
    x_train_raw = pd.read_csv('/Users/owenarink/Documents/University/neuralnets101/kaggle_commonsense/data/train_data.csv')

    # test_data_csv
    x_test_raw = pd.read_csv('/Users/owenarink/Documents/University/neuralnets101/kaggle_commonsense/data/test_data.csv') 
    
    # answer_csv
    y_train_raw = pd.read_csv('/Users/owenarink/Documents/University/neuralnets101/kaggle_commonsense/data/train_answers.csv')
    #print("answers columns:", y_train_raw.columns)
    #print("answers head:", y_train_raw.head())
    train_df, test_df = preprocess(x_train_raw, x_test_raw, y_train_raw)
    #print(train_df.shape)
    #print(test_df.shape)
    print('x_train_raw', x_train_raw)
    print('x_test_raw', x_test_raw)
    print('y_train_raw', y_train_raw)
    print('train_df', train_df)
    print('traindf:', train_df)
    print('testdf:', test_df)
    #print('x_train_preprocessed: ', train_df, 'y_train_preprocessed: ', test_df)
    #df = pd.concat([train_df, test_df], ignore_index=True)
    #train_set = df[df.answer != ]
    #test_set = df[df.answer == ]
    print("train_df columns: ", train_df.columns.tolist())
    print("test_df columns: ", test_df.columns.tolist())



