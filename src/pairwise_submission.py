import os
import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer

from src.processing import preprocess
from src.models.mlp_tfidf import MLP

OPTIONS = ["A", "B", "C"]

def build_pair_texts(df):
    texts, gids, opts = [], [], []
    for _, r in df.iterrows():
        s = str(r["FalseSent"])
        gid = r["id"]
        for opt in OPTIONS:
            texts.append(f"{s} [SEP] {str(r[f'Option{opt}'])}")
            gids.append(gid)
            opts.append(opt)
    return texts, np.array(gids), np.array(opts)

def main():
    x_train_raw = pd.read_csv("data/train_data.csv")
    x_test_raw  = pd.read_csv("data/test_data.csv")
    y_train_raw = pd.read_csv("data/train_answers.csv")

    train_df, test_df = preprocess(x_train_raw, x_test_raw, y_train_raw)

    # ---- Fit vectorizer on TRAIN PAIRWISE texts (do NOT densify) ----
    tr_texts, _, _ = build_pair_texts(train_df)
    vec = TfidfVectorizer(binary=True, ngram_range=(1, 2))
    vec.fit(tr_texts)

    # ---- Load model ----
    ckpt_path = "checkpoints/mlp_pairwise.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    n_features = len(vec.vocabulary_)
    model = MLP(n_inputs=n_features, n_hidden=[64], n_classes=1)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # ---- Build TEST PAIRWISE texts ----
    te_texts, te_gids, te_opts = build_pair_texts(test_df)

    # ---- Predict in batches (sparse -> dense per batch) ----
    batch_size = 512
    best = {}  # id -> (score, opt)

    with torch.no_grad():
        for start in range(0, len(te_texts), batch_size):
            end = min(start + batch_size, len(te_texts))
            X_sparse = vec.transform(te_texts[start:end])              # sparse
            X = X_sparse.toarray().astype(np.float32)                 # dense (small batch)

            logits = model(torch.tensor(X)).squeeze(1)
            scores = torch.sigmoid(logits).cpu().numpy()

            batch_gids = te_gids[start:end]
            batch_opts = te_opts[start:end]

            for s, gid, opt in zip(scores, batch_gids, batch_opts):
                if gid not in best or s > best[gid][0]:
                    best[gid] = (float(s), opt)

    submission = pd.DataFrame({
        "id": list(best.keys()),
        "answer": [best[i][1] for i in best.keys()]
    })

    out_path = "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path} with {len(submission)} rows.")
    print(submission.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
