from dependencies.dependencies import *
import os, math, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.bert_attention_experiments import BertSepDistanceTransformer
from src.tokenizer_bbpe import ensure_bbpe, load_bbpe, encode_grouped_bbpe

TOKENIZER_PATH = "checkpoints/transformer_attention_sepdist_tokenizer.json"
TOKENIZER_META_PATH = "checkpoints/transformer_attention_sepdist_tokenizer_meta.json"
HPARAMS_PATH = "checkpoints/transformer_attention_sepdist_hparams.json"


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05):
    def lr_lambda(step: int):
        step = min(step, total_steps)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
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
    model.eval()
    ce = nn.CrossEntropyLoss(label_smoothing=0.10)
    ds = TensorDataset(torch.tensor(X_va, dtype=torch.long), torch.tensor(y_va, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        B, K, L = xb.shape
        scores = model(xb.view(B * K, L)).view(B, K)
        loss = ce(scores, yb)
        total_loss += float(loss.item()) * B
        pred = torch.argmax(scores, dim=1)
        correct += (pred == yb).sum().item()
        total += B
        if device.type == "mps":
            torch.mps.empty_cache()
    return correct / max(1, total), total_loss / max(1, total)


def preprocess_transformer_grouped_bbpe(train_df: pd.DataFrame, seed=40, max_len=128, vocab_size=12000, min_freq=2):
    tr_rows, va_rows = train_test_split(train_df, test_size=0.2, random_state=seed, stratify=train_df["label"].astype(str))
    ensure_bbpe(train_df, tokenizer_path=TOKENIZER_PATH, meta_path=TOKENIZER_META_PATH, vocab_size=vocab_size, min_freq=min_freq)
    tok, meta = load_bbpe(TOKENIZER_PATH, TOKENIZER_META_PATH)
    pad_id = meta["pad_id"]
    unk_id = meta["unk_id"]
    sep_id = meta["sep_id"]
    X_tr = encode_grouped_bbpe(tr_rows, tok, max_len=max_len, pad_id=pad_id)
    X_va = encode_grouped_bbpe(va_rows, tok, max_len=max_len, pad_id=pad_id)
    mapping = {"A": 0, "B": 1, "C": 2}
    y_tr = np.array([mapping[x] for x in tr_rows["label"].astype(str).str.strip().values], dtype=np.int64)
    y_va = np.array([mapping[x] for x in va_rows["label"].astype(str).str.strip().values], dtype=np.int64)
    tok_info = {"pad_id": int(pad_id), "unk_id": int(unk_id), "sep_id": int(sep_id), "tokenizer_path": TOKENIZER_PATH, "tokenizer_meta_path": TOKENIZER_META_PATH, "vocab_size": int(tok.get_vocab_size()), "max_len": int(max_len), "seed": int(seed), "bbpe_vocab_size": int(vocab_size), "bbpe_min_freq": int(min_freq)}
    return X_tr, X_va, y_tr, y_va, tok_info


def train_grouped_transformer_attention_sepdist(train_df: pd.DataFrame, seed=40, max_len=128):
    device = get_device()
    print("Using device:", device)
    X_tr, X_va, y_tr, y_va, tok_info = preprocess_transformer_grouped_bbpe(train_df=train_df, seed=seed, max_len=max_len, vocab_size=12000, min_freq=2)
    pad_id, unk_id, sep_id, vocab_size = tok_info["pad_id"], tok_info["unk_id"], tok_info["sep_id"], tok_info["vocab_size"]
    print("------------------- Train Model Transformer Attention SepDist ----------------")
    print("X_tr:", X_tr.shape, X_tr.dtype)
    print("X_va:", X_va.shape, X_va.dtype)
    print("y_tr:", y_tr.shape, y_tr.dtype)
    print("y_va:", y_va.shape, y_va.dtype)
    print("vocab size:", vocab_size)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr).long(), torch.tensor(y_tr).long()), batch_size=32, shuffle=True)
    model_dim, num_heads, num_layers, ff_mult, dropout, pooling, max_len_pe = 256, 8, 4, 4, 0.30, "mean", 512
    model = BertSepDistanceTransformer(vocab_size=vocab_size, num_classes=1, pad_idx=pad_id, sep_idx=sep_id, model_dim=model_dim, num_heads=num_heads, num_layers=num_layers, ff_mult=ff_mult, dropout=dropout, max_len=max_len_pe, pooling=pooling).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    base_lr = 6e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5e-2)
    epochs, schedule_epochs = 120, 40
    total_steps = schedule_epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = make_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05)
    best_acc, best_loss, best_epoch_acc, best_epoch_loss, patience, bad = -1.0, float("inf"), -1, -1, 5, 0
    os.makedirs("checkpoints", exist_ok=True)
    hparams = {"pad_idx": int(pad_id), "sep_idx": int(sep_id), "model_dim": int(model_dim), "num_heads": int(num_heads), "num_layers": int(num_layers), "ff_mult": int(ff_mult), "dropout": float(dropout), "max_len": int(max_len_pe), "pooling": str(pooling), "bbpe_tokenizer_path": tok_info["tokenizer_path"], "tokenizer_meta_path": tok_info["tokenizer_meta_path"], "bbpe_vocab_size": int(tok_info["bbpe_vocab_size"]), "bbpe_min_freq": int(tok_info["bbpe_min_freq"]), "grouped_max_len": int(tok_info["max_len"]), "seed": int(tok_info["seed"])}
    with open(HPARAMS_PATH, "w") as f:
        json.dump(hparams, f)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch_x, batch_y in pbar:
            batch_x = token_dropout(batch_x.to(device), pad_id=pad_id, unk_id=unk_id, p=0.10)
            batch_y = batch_y.to(device)
            B, K, L = batch_x.shape
            scores = model(batch_x.view(B * K, L)).view(B, K)
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
        print(f"[GROUPED-CE] epoch [{epoch}/{epochs}] train_loss={train_loss:.4f} train_cat_acc={train_cat_acc:.4f} val_loss={val_loss:.4f} val_cat_acc={val_cat_acc:.4f} base_lr={base_lr:.2e} current_lr={scheduler.get_last_lr()[0]:.2e}")
        improved = False
        if val_cat_acc > best_acc + 1e-6:
            best_acc, best_epoch_acc = val_cat_acc, epoch
            torch.save(model.state_dict(), "checkpoints/transformer_attention_sepdist_best_acc.pt")
            improved = True
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch_loss = val_loss, epoch
            torch.save(model.state_dict(), "checkpoints/transformer_attention_sepdist_best_loss.pt")
            improved = True
        torch.save(model.state_dict(), "checkpoints/transformer_attention_sepdist.pt")
        if improved:
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}. Best acc={best_acc:.4f} (epoch {best_epoch_acc}), best loss={best_loss:.4f} (epoch {best_epoch_loss}).")
                break
    print("model completed")


if __name__ == "__main__":
    train_data = pd.read_csv("data/train_data.csv")
    labels = pd.read_csv("data/train_answers.csv")["answer"].astype(str).values
    train_data["label"] = labels
    train_grouped_transformer_attention_sepdist(train_data, seed=40, max_len=128)
