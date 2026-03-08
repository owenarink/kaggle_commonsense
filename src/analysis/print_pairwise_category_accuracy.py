import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.processing import preprocess
from src.models.mlp_tfidf import MLP

OPT = ["A", "B", "C"]

def build_pairwise_texts(df):
    texts = []
    group_ids = []
    opt_ids = []
    y = []

    for _, row in df.iterrows():
        s = str(row["FalseSent"])
        gid = row["id"]
        gold = str(row["label"]).strip()  # A/B/C

        for opt in OPT:
            r = str(row[f"Option{opt}"])
            texts.append(f"{s} [SEP] {r}")
            group_ids.append(gid)
            opt_ids.append(opt)
            y.append(1 if opt == gold else 0)

    return texts, np.array(y, dtype=np.float32), np.array(group_ids), np.array(opt_ids)

def regroup_argmax(scores, group_ids, opt_ids):
    best = {}
    for s, gid, opt in zip(scores, group_ids, opt_ids):
        if gid not in best or s > best[gid][0]:
            best[gid] = (s, opt)
    return {gid: opt for gid, (s, opt) in best.items()}

def category_accuracy(df, pred_by_id):
    correct = 0
    for _, row in df.iterrows():
        if pred_by_id[row["id"]] == row["label"]:
            correct += 1
    return correct / len(df)

def main():
    # Load
    x_train_raw = pd.read_csv("data/train_data.csv")
    x_test_raw  = pd.read_csv("data/test_data.csv")
    y_train_raw = pd.read_csv("data/train_answers.csv")
    train_df, _ = preprocess(x_train_raw, x_test_raw, y_train_raw)

    # Split by original rows
    train_rows, val_rows = train_test_split(
        train_df, test_size=0.2, random_state=40, stratify=train_df["label"].astype(str)
    )

    # Build pairwise texts
    tr_texts, tr_y, _, _ = build_pairwise_texts(train_rows)
    va_texts, va_y, va_gids, va_opts = build_pairwise_texts(val_rows)

    # TF-IDF on pairwise texts
    vec = TfidfVectorizer(binary=True, ngram_range=(1,2))
    Xtr = vec.fit_transform(tr_texts).toarray().astype(np.float32)
    Xva = vec.transform(va_texts).toarray().astype(np.float32)

    # Pairwise model: output 1 logit
    model = MLP(n_inputs=Xtr.shape[1], n_hidden=[64], n_classes=1)
    model.load_state_dict(torch.load("checkpoints/mlp_pairwise.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(Xva).float()).squeeze(1)
        scores = torch.sigmoid(logits).cpu().numpy()  # probability of "correct reason"

    pred_by_id = regroup_argmax(scores, va_gids, va_opts)
    acc = category_accuracy(val_rows, pred_by_id)

    print("Pairwise CategoryAccuracy =", acc)
    print("N =", len(val_rows))

if __name__ == "__main__":
    main()
