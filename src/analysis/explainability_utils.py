from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.models.mlp_tfidf import MLP
from src.processing import preprocess


ROOT = Path(__file__).resolve().parents[2]
LABELS = ["A", "B", "C"]


def load_train_test_frames():
    x_train_raw = pd.read_csv(ROOT / "data" / "train_data.csv")
    x_test_raw = pd.read_csv(ROOT / "data" / "test_data.csv")
    y_train_raw = pd.read_csv(ROOT / "data" / "train_answers.csv")
    return preprocess(x_train_raw, x_test_raw, y_train_raw)


def build_mlp_tfidf_bundle(seed: int = 40):
    train_df, test_df = load_train_test_frames()
    if "answer" not in train_df.columns:
        train_df = train_df.copy()
        train_df["answer"] = (
            train_df["FalseSent"].astype(str)
            + " [SEP] "
            + train_df["OptionA"].astype(str)
            + " [SEP] "
            + train_df["OptionB"].astype(str)
            + " [SEP] "
            + train_df["OptionC"].astype(str)
        )
    vectorizer = TfidfVectorizer(binary=True, ngram_range=(1, 2), max_features=20000, min_df=2)
    x_all = vectorizer.fit_transform(train_df["answer"]).toarray().astype(np.float32)
    mapping = {"A": 0, "B": 1, "C": 2}
    y_all = np.array([mapping[label] for label in train_df["label"].astype(str).str.strip().values], dtype=np.int64)
    x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.2, random_state=seed)
    return {
        "x_train": x_train,
        "x_val": x_val,
        "y_train": y_train,
        "y_val": y_val,
        "vectorizer": vectorizer,
        "feature_names": vectorizer.get_feature_names_out(),
        "train_df": train_df,
    }


def load_mlp_model(input_dim: int):
    model = MLP(n_inputs=input_dim, n_hidden=[64], n_classes=3)
    ckpt = ROOT / "checkpoints" / "mlp_tfidf.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


def predict_mlp_proba(model: MLP, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

