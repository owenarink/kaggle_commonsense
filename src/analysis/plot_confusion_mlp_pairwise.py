import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.processing import preprocess
from src.models.mlp_tfidf import MLP

OPTIONS = ["A", "B", "C"]
OPT2IDX = {"A": 0, "B": 1, "C": 2}
IDX2OPT = {0: "A", 1: "B", 2: "C"}

def build_pairwise(df):
    texts = []
    group_ids = []
    opt_ids = []
    gold_ids = []
    for _, r in df.iterrows():
        s = str(r["FalseSent"])
        gid = r["id"]
        gold = str(r["label"]).strip()
        for opt in OPTIONS:
            texts.append(f"{s} [SEP] {str(r[f'Option{opt}'])}")
            group_ids.append(gid)
            opt_ids.append(opt)
            gold_ids.append(gold)
    return texts, np.array(group_ids), np.array(opt_ids), np.array(gold_ids)

def confusion_matrix(y_true, y_pred, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def per_class_accuracy(cm):
    denom = cm.sum(axis=1)
    out = np.zeros(cm.shape[0], dtype=np.float64)
    for i in range(cm.shape[0]):
        out[i] = (cm[i, i] / denom[i]) if denom[i] > 0 else 0.0
    return out

def main():
    import pandas as pd
    
    os.makedirs("outputs/plots", exist_ok=True)  # <-- ADD THIS HERE

    x_train_raw = pd.read_csv("data/train_data.csv")
    x_test_raw  = pd.read_csv("data/test_data.csv")
    y_train_raw = pd.read_csv("data/train_answers.csv")

    train_df, _ = preprocess(x_train_raw, x_test_raw, y_train_raw)

    # Use the same split strategy as training (split by original rows)
    _, val_rows = train_test_split(
        train_df, test_size=0.2, random_state=40, stratify=train_df["label"].astype(str)
    )

    va_texts, va_gids, va_opts, va_gold = build_pairwise(val_rows)

    # TF-IDF transform
    # Important: to match training, we must fit on TRAIN rows' pair texts too
    train_rows, _ = train_test_split(
        train_df, test_size=0.2, random_state=40, stratify=train_df["label"].astype(str)
    )
    tr_texts, _, _, _ = build_pairwise(train_rows)

    vec = TfidfVectorizer(binary=True, ngram_range=(1, 2))
    Xtr = vec.fit_transform(tr_texts).toarray().astype(np.float32)
    Xva = vec.transform(va_texts).toarray().astype(np.float32)

    # Load pairwise model
    ckpt_path = "checkpoints/mlp_pairwise.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model = MLP(n_inputs=Xtr.shape[1], n_hidden=[64], n_classes=1)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # Predict pair scores
    with torch.no_grad():
        logits = model(torch.tensor(Xva).float()).squeeze(1)
        scores = torch.sigmoid(logits).cpu().numpy()

    # Regroup by id: pick best option
    best = {}
    for s, gid, opt, gold in zip(scores, va_gids, va_opts, va_gold):
        if gid not in best or s > best[gid][0]:
            best[gid] = (s, opt, gold)

    y_true = []
    y_pred = []
    for gid, (_, opt_pred, gold) in best.items():
        y_true.append(OPT2IDX[gold])
        y_pred.append(OPT2IDX[opt_pred])

    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, n_classes=3)
    per_acc = per_class_accuracy(cm)

    print("Pairwise category accuracy (val):", acc)
    print("Per-class accuracy:", {IDX2OPT[i]: float(per_acc[i]) for i in range(3)})

    # ---- Plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Pairwise Confusion Matrix (val acc={acc:.3f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1, 2], ["A", "B", "C"])
    plt.yticks([0, 1, 2], ["A", "B", "C"])

    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig("outputs/plots/pairwise_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ---- Plot per-class accuracy
    plt.figure(figsize=(6, 4))
    plt.bar(["A", "B", "C"], per_acc)
    plt.ylim(0, 1)
    plt.title("Per-class Accuracy (Pairwise Validation)")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("outputs/plots/pairwise_per_class_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
