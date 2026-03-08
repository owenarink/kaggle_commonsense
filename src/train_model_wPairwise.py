from src.models.mlp_tfidf import MLP
from src.processing import preprocess
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.eval import evaluate_model  # your existing multiclass evaluator


# -------------------------
# Multiclass (your current)
# -------------------------

def load_prep_data_multiclass():
    x_train_raw = pd.read_csv("data/train_data.csv")
    x_test_raw = pd.read_csv("data/test_data.csv")
    y_train_raw = pd.read_csv("data/train_answers.csv")

    train_df, test_df = preprocess(x_train_raw, x_test_raw, y_train_raw)

    # your existing get_tfidf class returns x_train, x_val, y_train, y_val
    from src.features.tfidf import get_tfidf
    x_train, x_val, y_train, y_val = get_tfidf(train_df, test_df)

    # ---- SAME PRINTS AS BEFORE (multiclass) ----
    print('------------------- TRAIN MODEL ----------------')
    print('train_model, after preprocess and get_tfidf: x_train', x_train)
    print('train_model, after preprocess and get_tfidf: x_val', x_val)
    print('train_model, after preprocess and get_tfidf: y_train', y_train)
    print('train_model, after preprocess and get_tfidf: y_val', y_val)

    return x_train, x_val, y_train, y_val


def train_multiclass(x_train, x_val, y_train, y_val):
    train_dataset = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).long())
    val_dataset = TensorDataset(torch.tensor(x_val).float(), torch.tensor(y_val).long())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = MLP(n_inputs=x_train.shape[1], n_hidden=[64], n_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_crit = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            loss = loss_crit(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = evaluate_model(model, train_loader)
        val_acc = evaluate_model(model, val_loader)
        print(f"[MULTI] epoch [{epoch+1}/{epochs}] loss={loss.item():.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/mlp_tfidf.pt")
    print("Saved multiclass model to checkpoints/mlp_tfidf.pt")


# -------------------------
# Pairwise (new)
# -------------------------

OPTIONS = ["A", "B", "C"]


def build_pairwise_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        s = str(r["FalseSent"])
        gid = r["id"]
        gold = str(r["label"]).strip()
        for opt in OPTIONS:
            rows.append({
                "group_id": gid,
                "opt": opt,
                "text": f"{s} [SEP] {str(r[f'Option{opt}'])}",
                "y": 1.0 if opt == gold else 0.0,
                "gold": gold,
            })
    return pd.DataFrame(rows)


def load_prep_data_pairwise(seed=40):
    x_train_raw = pd.read_csv("data/train_data.csv")
    x_test_raw = pd.read_csv("data/test_data.csv")
    y_train_raw = pd.read_csv("data/train_answers.csv")

    train_df, _ = preprocess(x_train_raw, x_test_raw, y_train_raw)

    train_rows, val_rows = train_test_split(
        train_df,
        test_size=0.2,
        random_state=seed,
        stratify=train_df["label"].astype(str)
    )

    train_pairs = build_pairwise_df(train_rows)
    val_pairs = build_pairwise_df(val_rows)

    vec = TfidfVectorizer(binary=True, ngram_range=(1, 2))
    X_tr = vec.fit_transform(train_pairs["text"]).toarray().astype(np.float32)
    X_va = vec.transform(val_pairs["text"]).toarray().astype(np.float32)

    y_tr = train_pairs["y"].values.astype(np.float32)
    y_va = val_pairs["y"].values.astype(np.float32)

    va_group_ids = val_pairs["group_id"].values
    va_opts = val_pairs["opt"].values
    va_gold = val_rows.set_index("id")["label"].to_dict()

    # ---- SAME PRINTS AS BEFORE (pairwise) ----
    print('------------------- TRAIN PAIRWISE MODEL ----------------')
    print('train_model, pairwise after preprocess and tfidf: X_tr', X_tr)
    print('train_model, pairwise after preprocess and tfidf: X_va', X_va)
    print('train_model, pairwise after preprocess and tfidf: y_tr', y_tr)
    print('train_model, pairwise after preprocess and tfidf: y_va', y_va)

    return X_tr, X_va, y_tr, y_va, va_group_ids, va_opts, va_gold


def pairwise_accuracy_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    return (preds.view(-1) == y_true.long().view(-1)).float().mean().item()


def category_accuracy_from_pair_scores(scores: np.ndarray, group_ids: np.ndarray, opts: np.ndarray, gold_by_id: dict) -> float:
    best = {}
    for s, gid, opt in zip(scores, group_ids, opts):
        if gid not in best or s > best[gid][0]:
            best[gid] = (s, opt)

    correct = 0
    total = 0
    for gid, (_, pred_opt) in best.items():
        total += 1
        if pred_opt == gold_by_id[gid]:
            correct += 1

    return correct / total if total > 0 else 0.0


def train_pairwise(X_tr, X_va, y_tr, y_va, va_group_ids, va_opts, va_gold):
    train_dataset = TensorDataset(torch.tensor(X_tr).float(), torch.tensor(y_tr).float())
    val_dataset = TensorDataset(torch.tensor(X_va).float(), torch.tensor(y_va).float())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = MLP(n_inputs=X_tr.shape[1], n_hidden=[64], n_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_crit = nn.BCEWithLogitsLoss()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            logits = model(batch_x).view(-1)
            loss = loss_crit(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        all_logits = []
        with torch.no_grad():
            for bx, _ in val_loader:
                lg = model(bx).view(-1)
                all_logits.append(lg.cpu())
        all_logits = torch.cat(all_logits, dim=0)

        val_pair_acc = pairwise_accuracy_from_logits(all_logits, torch.tensor(y_va))
        val_scores = torch.sigmoid(all_logits).numpy()
        val_cat_acc = category_accuracy_from_pair_scores(val_scores, va_group_ids, va_opts, va_gold)

        print(f"[PAIR] epoch [{epoch+1}/{epochs}] loss={loss.item():.4f} val_pair_acc={val_pair_acc:.4f} val_cat_acc={val_cat_acc:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/mlp_pairwise.pt")
    print("Saved pairwise model to checkpoints/mlp_pairwise.pt")


if __name__ == "__main__":
    print("=== MULTICLASS TRAINING ===")
    x_train, x_val, y_train, y_val = load_prep_data_multiclass()
    train_multiclass(x_train, x_val, y_train, y_val)

    print("\n=== PAIRWISE TRAINING ===")
    X_tr, X_va, y_tr, y_va, va_group_ids, va_opts, va_gold = load_prep_data_pairwise(seed=40)
    train_pairwise(X_tr, X_va, y_tr, y_va, va_group_ids, va_opts, va_gold)
