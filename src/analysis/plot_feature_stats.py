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

def main():
    import pandas as pd
    x_train_raw = pd.read_csv("data/train_data.csv")
    x_test_raw  = pd.read_csv("data/test_data.csv")
    y_train_raw = pd.read_csv("data/train_answers.csv")
    train_df, test_df = preprocess(x_train_raw, x_test_raw, y_train_raw)

    X_tr, X_va, y_tr, y_va = get_tfidf(train_df, test_df)

    # ---- Feature sparsity stats (dense arrays)
    tr_nz = np.count_nonzero(X_tr, axis=1)
    va_nz = np.count_nonzero(X_va, axis=1)

    plt.figure(figsize=(7, 4))
    plt.hist(tr_nz, bins=50, alpha=0.7, label="train")
    plt.hist(va_nz, bins=50, alpha=0.7, label="val")
    plt.title("Non-zero TF-IDF features per sample")
    plt.xlabel("# non-zero features")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/non_zero_tfidf_features_per_sample_mlp.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ---- Load model and inspect weight distributions
    model = MLP(n_inputs=X_tr.shape[1], n_hidden=[64], n_classes=3)
    ckpt_path = "checkpoints/mlp_tfidf.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # Try to find the first Linear layer's weight tensor
    # This works for typical Sequential models: [Linear, ReLU, Linear]
    first_w = None
    for p_name, p in model.named_parameters():
        if "weight" in p_name:
            first_w = p.detach().cpu().numpy().ravel()
            break
    if first_w is None:
        raise RuntimeError("Could not find a weight parameter in the model.")

    plt.figure(figsize=(7, 4))
    plt.hist(first_w, bins=60)
    plt.title("Histogram of first weight tensor values")
    plt.xlabel("weight value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("outputs/plots/Histogram_first_weight_tensors_mlp.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
