import torch
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset

from src.processing import preprocess
from src.features.tfidf import get_tfidf
from src.models.mlp_tfidf import MLP


IDX2LABEL = {0: "A", 1: "B", 2: "C"}


def category_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    CategoryAccuracy = (1/N) * sum_i 1[y_i == yhat_i]
    """
    y_true = y_true.view(-1).long()
    y_pred = y_pred.view(-1).long()
    return (y_true == y_pred).float().mean().item()


def main():
    # Load and preprocess
    x_train_raw = pd.read_csv("data/train_data.csv")
    x_test_raw  = pd.read_csv("data/test_data.csv")
    y_train_raw = pd.read_csv("data/train_answers.csv")
    train_df, test_df = preprocess(x_train_raw, x_test_raw, y_train_raw)

    # Recreate TF-IDF split (same as training)
    x_train, x_val, y_train, y_val = get_tfidf(train_df, test_df)

    # Load trained model
    model = MLP(n_inputs=x_train.shape[1], n_hidden=[64], n_classes=3)
    model.load_state_dict(torch.load("checkpoints/mlp_tfidf.pt", map_location="cpu"))
    model.eval()

    # Predict on validation set
    Xv = torch.tensor(x_val).float()
    yv = torch.tensor(y_val).long()

    with torch.no_grad():
        logits = model(Xv)
        preds = torch.argmax(logits, dim=1)

    acc = category_accuracy(yv, preds)

    print("CategoryAccuracy =", acc)
    print(f"N = {len(yv)}")
    print("Example predictions (first 10):")
    for i in range(min(10, len(yv))):
        print(f"  true={IDX2LABEL[int(yv[i])]}  pred={IDX2LABEL[int(preds[i])]}")

if __name__ == "__main__":
    main()
