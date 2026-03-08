import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.processing import preprocess
from src.models.mlp_tfidf import MLP

OPTIONS = ["A", "B", "C"]

def build_pairwise_df(df):
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
            })
    return rows

def main():
    import pandas as pd
    
    os.makedirs("outputs/plots", exist_ok=True)

    x_train_raw = pd.read_csv("data/train_data.csv")
    x_test_raw  = pd.read_csv("data/test_data.csv")
    y_train_raw = pd.read_csv("data/train_answers.csv")

    train_df, _ = preprocess(x_train_raw, x_test_raw, y_train_raw)

    train_rows, val_rows = train_test_split(
        train_df, test_size=0.2, random_state=40, stratify=train_df["label"].astype(str)
    )

    tr_pairs = build_pairwise_df(train_rows)
    va_pairs = build_pairwise_df(val_rows)

    tr_texts = [r["text"] for r in tr_pairs]
    va_texts = [r["text"] for r in va_pairs]

    vec = TfidfVectorizer(binary=True, ngram_range=(1, 2))
    Xtr = vec.fit_transform(tr_texts).toarray().astype(np.float32)
    Xva = vec.transform(va_texts).toarray().astype(np.float32)

    # ---- Feature sparsity stats (dense arrays)
    tr_nz = np.count_nonzero(Xtr, axis=1)
    va_nz = np.count_nonzero(Xva, axis=1)

    plt.figure(figsize=(7, 4))
    plt.hist(tr_nz, bins=50, alpha=0.7, label="train (pairs)")
    plt.hist(va_nz, bins=50, alpha=0.7, label="val (pairs)")
    plt.title("Non-zero TF-IDF features per pair sample")
    plt.xlabel("# non-zero features")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/feature_stats_mlp_pairwise.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ---- Load pairwise model and inspect weight distributions
    ckpt_path = "checkpoints/mlp_pairwise.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model = MLP(n_inputs=Xtr.shape[1], n_hidden=[64], n_classes=1)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # Find first weight tensor
    first_w = None
    for name, p in model.named_parameters():
        if "weight" in name:
            first_w = p.detach().cpu().numpy().ravel()
            break
    if first_w is None:
        raise RuntimeError("Could not find a weight parameter in the model.")

    plt.figure(figsize=(7, 4))
    plt.hist(first_w, bins=60)
    plt.title("Histogram of first weight tensor values (pairwise MLP)")
    plt.xlabel("weight value")
    plt.ylabel("count")
    plt.tight_layout()


if __name__ == "__main__":
    main()
