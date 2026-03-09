from dependencies.dependencies import *

OPTIONS = ["A", "B", "C"]

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


def load_prep_data_pairwise(seed=40, max_len=128, min_freq=2, max_vocab=50000):
    x_train_raw = pd.read_csv("data/train_data.csv")
    x_test_raw  = pd.read_csv("data/test_data.csv")
    y_train_raw = pd.read_csv("data/train_answers.csv")

    train_df, _ = preprocess(x_train_raw, x_test_raw, y_train_raw)

    from src.processing import preprocess_cnn
    X_tr, X_va, y_tr, y_va, vocab, va_group_ids, va_opts, va_gold = preprocess_cnn(
        train_df=train_df,
        seed=seed,
        mode="pairwise",
        max_len=max_len,
        min_freq=min_freq,
        max_vocab=max_vocab,
    )
    print("y_tr mean (should be ~0.333):", float(np.mean(y_tr)))
    print("first 6 y_tr:", y_tr[:6])
    print('------------------- TRAIN PAIRWISE TRANSFORMER MODEL ----------------')
    print('X_tr:', np.asarray(X_tr).shape, np.asarray(X_tr).dtype)
    print('X_va:', np.asarray(X_va).shape, np.asarray(X_va).dtype)
    print('y_tr:', np.asarray(y_tr).shape, np.asarray(y_tr).dtype)
    print('y_va:', np.asarray(y_va).shape, np.asarray(y_va).dtype)
    print('vocab size:', len(vocab))

    return X_tr, X_va, y_tr, y_va, va_group_ids, va_opts, va_gold, vocab


def train_pairwise_transformer(X_tr, X_va, y_tr, y_va, va_group_ids, va_opts, va_gold, vocab):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = TensorDataset(torch.tensor(X_tr).long(), torch.tensor(y_tr).float())
    val_dataset   = TensorDataset(torch.tensor(X_va).long(), torch.tensor(y_va).float())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

    from src.models.transformer import TextTransformer
    model = TextTransformer(
        vocab_size=len(vocab),
        num_classes=1,
        pad_idx=0,
        model_dim=128,
        num_heads=4,
        num_layers=2,
        ff_mult=2,
        dropout=0.1,
        max_len=512,
        pooling="mean",
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_crit = nn.BCEWithLogitsLoss()

    epochs = 20

    total_steps = epochs * len(train_loader)
    total_pbar = tqdm(total=total_steps, desc="Total", unit="batch")

    for epoch in range(epochs):
        model.train()
        epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)

        for batch_x, batch_y in epoch_pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x).view(-1)
            loss = loss_crit(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_pbar.set_postfix(loss=f"{loss.item():.4f}")
            total_pbar.update(1)

        epoch_pbar.close()

        model.eval()
        all_logits = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                lg = model(bx).view(-1).cpu()
                all_logits.append(lg)
        all_logits = torch.cat(all_logits, dim=0)

        val_pair_acc = pairwise_accuracy_from_logits(all_logits, torch.tensor(y_va))
        val_scores = torch.sigmoid(all_logits).numpy()
        val_cat_acc = category_accuracy_from_pair_scores(val_scores, va_group_ids, va_opts, va_gold)

        print(f"[PAIR-TRANSFORMER] epoch [{epoch+1}/{epochs}] loss={loss.item():.4f} val_pair_acc={val_pair_acc:.4f} val_cat_acc={val_cat_acc:.4f}")

    total_pbar.close()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/transformer_pairwise.pt")
    print("Saved pairwise Transformer model to checkpoints/transformer_pairwise.pt")


if __name__ == "__main__":
    X_tr, X_va, y_tr, y_va, va_group_ids, va_opts, va_gold, vocab = load_prep_data_pairwise(seed=40)
    train_pairwise_transformer(X_tr, X_va, y_tr, y_va, va_group_ids, va_opts, va_gold, vocab)
    print('train_model_transformer_pairwise completed\n')
    print('------------------------------------------------')
