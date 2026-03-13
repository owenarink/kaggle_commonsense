
from dependencies.dependencies import *
import os, math, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.transformer import TextTransformer
from src.processing import preprocess

from src.tokenizer_bbpe import ensure_bbpe, load_bbpe, encode_grouped_bbpe


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def token_dropout(x_ids: torch.Tensor, pad_id: int, unk_id: int, p: float):
    if p <= 0:
        return x_ids
    x = x_ids.clone()
    mask = (x != pad_id) & (torch.rand_like(x.float()) < p)
    x[mask] = unk_id
    return x

@torch.no_grad()
def grouped_val_metrics(model, X_va, y_va, device, batch_size=128):
    """
    Returns: (val_cat_acc, val_loss)
    """
    model.eval()
    val_crit = nn.CrossEntropyLoss()  # no label smoothing for metric reporting

    ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.long),
        torch.tensor(y_va, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    total = 0
    correct = 0
    loss_sum = 0.0

    for xb, yb in loader:
        xb = xb.to(device)  # [B,3,L]
        yb = yb.to(device)  # [B]

        B, K, L = xb.shape
        scores = model(xb.view(B * K, L)).view(B, K)   # [B,3]
        loss = val_crit(scores, yb)

        pred = torch.argmax(scores, dim=1)
        correct += (pred == yb).sum().item()
        total += B
        loss_sum += float(loss.item()) * B

        if device.type == "mps":
            torch.mps.empty_cache()

    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    return acc, avg_loss


def preprocess_transformer_grouped_bbpe(train_df: pd.DataFrame, seed=40, max_len=128, vocab_size=12000, min_freq=2):
    # split by question (not by pairs)
    tr_rows, va_rows = train_test_split(
        train_df,
        test_size=0.2,
        random_state=seed,
        stratify=train_df["label"].astype(str),
    )

    # make sure tokenizer exists (train it once). You can train on FULL train_df for best tokenizer coverage.
    ensure_bbpe(train_df, tokenizer_path="checkpoints/bbpe_tokenizer.json", vocab_size=vocab_size, min_freq=min_freq)
    tok, pad_id, unk_id = load_bbpe("checkpoints/bbpe_tokenizer.json")

    # encode grouped: [N,3,L]
    X_tr = encode_grouped_bbpe(tr_rows, tok, max_len=max_len, pad_id=pad_id)
    X_va = encode_grouped_bbpe(va_rows, tok, max_len=max_len, pad_id=pad_id)

    # labels: A/B/C -> 0/1/2
    mapping = {"A": 0, "B": 1, "C": 2}
    y_tr = np.array([mapping[x] for x in tr_rows["label"].astype(str).str.strip().values], dtype=np.int64)
    y_va = np.array([mapping[x] for x in va_rows["label"].astype(str).str.strip().values], dtype=np.int64)

    tok_info = {
        "pad_id": int(pad_id),
        "unk_id": int(unk_id),
        "tokenizer_path": "checkpoints/bbpe_tokenizer.json",
        "vocab_size": int(tok.get_vocab_size()),
        "max_len": int(max_len),
        "seed": int(seed),
        "bbpe_vocab_size": int(vocab_size),
        "bbpe_min_freq": int(min_freq),
    }
    return X_tr, X_va, y_tr, y_va, tok_info


def train_grouped_transformer_bbpe(train_df: pd.DataFrame, seed=40, max_len=128):
    device = get_device()
    print("Using device:", device)

    X_tr, X_va, y_tr, y_va, tok_info = preprocess_transformer_grouped_bbpe(
        train_df=train_df,
        seed=seed,
        max_len=max_len,
        vocab_size=12000,
        min_freq=2,
    )
    pad_id = tok_info["pad_id"]
    unk_id = tok_info["unk_id"]
    vocab_size = tok_info["vocab_size"]

    print("------------------- TRAIN GROUPED TRANSFORMER MODEL ----------------")
    print("X_tr:", X_tr.shape, X_tr.dtype)
    print("X_va:", X_va.shape, X_va.dtype)
    print("y_tr:", y_tr.shape, y_tr.dtype)
    print("y_va:", y_va.shape, y_va.dtype)
    print("vocab size:", vocab_size)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr).long(), torch.tensor(y_tr).long()),
        batch_size=32,
        shuffle=True,
    )

    model_dim = 256
    num_heads = 8
    num_layers = 4
    ff_mult = 4
    dropout = 0.35
    pooling = "cls"
    max_len_pe = 512

    model = TextTransformer(
        vocab_size=vocab_size,
        num_classes=1,         # scorer per option => outputs [B*3,1]
        pad_idx=pad_id,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_mult=ff_mult,
        dropout=dropout,
        max_len=max_len_pe,
        pooling=pooling,
    ).to(device)

    # Grouped CE over [B,3]
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)

    base_lr = 6e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=3e-2)

    epochs = 80
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = make_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.2)

    best = -1.0
    best_epoch = -1
    patience = 10
    bad = 0

    os.makedirs("checkpoints", exist_ok=True)

    # save hparams for submits.py (so you reconstruct exactly)
    hparams = {
        "pad_idx": int(pad_id),
        "model_dim": int(model_dim),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "ff_mult": int(ff_mult),
        "dropout": float(dropout),
        "max_len": int(max_len_pe),
        "pooling": str(pooling),
        "bbpe_tokenizer_path": tok_info["tokenizer_path"],
        "bbpe_vocab_size": int(tok_info["bbpe_vocab_size"]),
        "bbpe_min_freq": int(tok_info["bbpe_min_freq"]),
        "grouped_max_len": int(tok_info["max_len"]),
        "seed": int(tok_info["seed"]),
    }
    with open("checkpoints/transformer_pairwise_hparams.json", "w") as f:
        json.dump(hparams, f)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)  # [B,3,L]
            batch_y = batch_y.to(device)  # [B]

            # (optional) keep your token_dropout here if you use it
            batch_x = token_dropout(batch_x, pad_id=pad_id, unk_id=unk_id, p=0.10)

            B, K, L = batch_x.shape
            x_flat = batch_x.view(B * K, L)

            scores = model(x_flat).view(B, K)   # [B,3]i
            loss = criterion(scores, batch_y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step() 
           
            running_loss += float(loss.item()) * B
            pred = torch.argmax(scores, dim=1)
            train_correct += (pred == batch_y).sum().item()
            train_total += B

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = running_loss / max(train_total, 1)
        train_cat_acc = train_correct / max(train_total, 1)
        val_cat_acc, val_loss = grouped_val_metrics(model, X_va, y_va, device, batch_size=128)

        print(
            f"[GROUPED-CE] epoch [{epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} train_cat_acc={train_cat_acc:.4f} "
            f"val_loss={val_loss:.4f} val_cat_acc={val_cat_acc:.4f} "
            f"base_lr:={base_lr:.4f} "
            f"current_lr={scheduler.get_last_lr()[0]:.2e} "
        )
        if val_cat_acc > best + 1e-6:
            best = val_cat_acc
            best_epoch = epoch
            bad = 0
            torch.save(model.state_dict(), "checkpoints/transformer_pairwise_best.pt")
            torch.save(model.state_dict(), "checkpoints/transformer_pairwise.pt")
            print(f"Saved BEST -> checkpoints/transformer_pairwise_best.pt (val_cat_acc={best:.4f})")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}. Best was {best:.4f} at epoch {best_epoch}.")
                break



if __name__ == "__main__":
    x_train_raw = pd.read_csv("data/train_data.csv")
    x_test_raw  = pd.read_csv("data/test_data.csv")
    y_train_raw = pd.read_csv("data/train_answers.csv")

    train_df, _ = preprocess(x_train_raw, x_test_raw, y_train_raw)

    train_grouped_transformer_bbpe(train_df, seed=40, max_len=128)
    print("train_model_transformer (grouped-bbpe) completed\n")
    print("------------------------------------------------")
