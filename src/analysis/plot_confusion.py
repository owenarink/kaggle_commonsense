import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch

from src.processing import preprocess
from src.features.tfidf import get_tfidf
from src.models.mlp_tfidf import MLP

LABELS = ["A", "B", "C"]

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    # diag / row-sum
    denom = cm.sum(axis=1)
    acc = np.zeros(cm.shape[0], dtype=np.float64)
    for i in range(cm.shape[0]):
        acc[i] = (cm[i, i] / denom[i]) if denom[i] > 0 else 0.0
    return acc

def main():
    # Load data
    import pandas as pd

    os.makedirs("outputs/plots", exist_ok=True)

    x_train_raw = pd.read_csv("data/train_data.csv")
    x_test_raw  = pd.read_csv("data/test_data.csv")
    y_train_raw = pd.read_csv("data/train_answers.csv")
    train_df, test_df = preprocess(x_train_raw, x_test_raw, y_train_raw)

    # Recreate TF-IDF split + labels (seed inside get_tfidf)
    X_tr, X_va, y_tr, y_va = get_tfidf(train_df, test_df)

    # Load model
    model = MLP(n_inputs=X_tr.shape[1], n_hidden=[64], n_classes=3)
    ckpt_path = "checkpoints/mlp_tfidf.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # Predict on val
    with torch.no_grad():
        logits = model(torch.tensor(X_va).float())
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    # Confusion matrix + accuracies
    cm = confusion_matrix(y_va, y_pred, n_classes=3)
    acc = (y_pred == y_va).mean()
    per_acc = per_class_accuracy(cm)

    print("Val accuracy:", float(acc))
    print("Per-class accuracy:", dict(zip(LABELS, per_acc.tolist())))

    # ---- Plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix (val acc={acc:.3f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0,1,2], LABELS)
    plt.yticks([0,1,2], LABELS)

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig("outputs/plots/plot_confusion_mlp.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ---- Plot per-class accuracy
    plt.figure(figsize=(6, 4))
    plt.bar(LABELS, per_acc)
    plt.ylim(0, 1)
    plt.title("Per-class Accuracy (Validation)")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("outputs/plots/plot_per_class_accuracy_mlp.png", dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
