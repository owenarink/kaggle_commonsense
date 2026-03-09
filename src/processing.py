import numpy as np
import csv 
import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split

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

OPTIONS = ["A", "B", "C"]

def simple_tokenize(text: str):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\s]", text.lower())

def build_vocab(texts, min_freq=2, max_vocab=50000):
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))

    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, freq in counter.most_common():
        if freq < min_freq:
            break
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
        if len(vocab) >= max_vocab:
            break
    return vocab

def encode_text(text: str, vocab: dict, max_len: int):
    toks = simple_tokenize(text)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in toks][:max_len]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids

def make_text_multiclass(df: pd.DataFrame):
    return (
        df["FalseSent"].astype(str) + " [SEP] " +
        df["OptionA"].astype(str) + " [SEP] " +
        df["OptionB"].astype(str) + " [SEP] " +
        df["OptionC"].astype(str)
    ).tolist()

def make_pairwise_rows(df: pd.DataFrame):
    rows = []
    for _, r in df.iterrows():
        s = str(r["FalseSent"])
        gold = str(r["label"]).strip()
        gid = r["id"]
        for opt in OPTIONS:
            rows.append({
                "group_id": gid,
                "opt": opt,
                "text": f"{s} [SEP] {str(r[f'Option{opt}'])}",
                "y": 1 if opt == gold else 0
            })
    return pd.DataFrame(rows)

def preprocess_cnn(train_df: pd.DataFrame, seed=40, mode="multiclass", max_len=128, min_freq=2, max_vocab=50000):
    if mode == "multiclass":
        tr_rows, va_rows = train_test_split(
            train_df, test_size=0.2, random_state=seed, stratify=train_df["label"].astype(str)
        )

        tr_texts = make_text_multiclass(tr_rows)
        va_texts = make_text_multiclass(va_rows)

        vocab = build_vocab(tr_texts, min_freq=min_freq, max_vocab=max_vocab)

        X_tr = np.array([encode_text(t, vocab, max_len) for t in tr_texts], dtype=np.int64)
        X_va = np.array([encode_text(t, vocab, max_len) for t in va_texts], dtype=np.int64)

        mapping = {"A": 0, "B": 1, "C": 2}
        y_tr = np.array([mapping[x] for x in tr_rows["label"].astype(str).str.strip().values], dtype=np.int64)
        y_va = np.array([mapping[x] for x in va_rows["label"].astype(str).str.strip().values], dtype=np.int64)

        return X_tr, X_va, y_tr, y_va, vocab

    if mode == "pairwise":
        tr_rows, va_rows = train_test_split(
            train_df, test_size=0.2, random_state=seed, stratify=train_df["label"].astype(str)
        )

        tr_pairs = make_pairwise_rows(tr_rows)
        va_pairs = make_pairwise_rows(va_rows)

        vocab = build_vocab(tr_pairs["text"].tolist(), min_freq=min_freq, max_vocab=max_vocab)

        X_tr = np.array([encode_text(t, vocab, max_len) for t in tr_pairs["text"].tolist()], dtype=np.int64)
        X_va = np.array([encode_text(t, vocab, max_len) for t in va_pairs["text"].tolist()], dtype=np.int64)

        y_tr = tr_pairs["y"].values.astype(np.float32)
        y_va = va_pairs["y"].values.astype(np.float32)

        va_group_ids = va_pairs["group_id"].values
        va_opts = va_pairs["opt"].values
        va_gold = va_rows.set_index("id")["label"].to_dict()

        return X_tr, X_va, y_tr, y_va, vocab, va_group_ids, va_opts, va_gold

    raise ValueError("mode must be 'multiclass' or 'pairwise'")

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

    print('PROCESSING COMPLETED\n')
    print('------------------------------------------------')

