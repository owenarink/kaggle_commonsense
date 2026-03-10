from dependencies.dependencies import *
import json, os

IDX2LBL = {0: "A", 1: "B", 2: "C"}
OPTIONS = ["A", "B", "C"]


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def infer_expected_in_dim(ckpt_path: str) -> int:
    sd = torch.load(ckpt_path, map_location="cpu")
    for k in ("layers.0.fc.weight", "layers.0.weight"):
        if k in sd:
            return int(sd[k].shape[1])
    for k, v in sd.items():
        if k.endswith("weight") and hasattr(v, "shape") and len(v.shape) == 2:
            return int(v.shape[1])
    raise RuntimeError(f"Could not infer input dim from checkpoint: {ckpt_path}")


def infer_transformer_hparams(ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")
    if "embedding.weight" not in sd:
        raise RuntimeError("Could not infer transformer dims: missing embedding.weight in checkpoint")

    vocab_size, model_dim = sd["embedding.weight"].shape[0], sd["embedding.weight"].shape[1]

    layer_ids = set()
    for k in sd.keys():
        if k.startswith("transformer.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_ids.add(int(parts[2]))
    num_layers = (max(layer_ids) + 1) if layer_ids else 0
    if num_layers <= 0:
        raise RuntimeError("Could not infer transformer num_layers from checkpoint keys")

    ff_w_key = f"transformer.layers.0.ff.0.weight"
    ff_mult = 2
    if ff_w_key in sd:
        ff_dim = int(sd[ff_w_key].shape[0])
        ff_mult = int(round(ff_dim / float(model_dim)))

    candidates = [8, 4, 2, 1, 16]
    num_heads = None
    for h in candidates:
        if model_dim % h == 0:
            num_heads = h
            break
    if num_heads is None:
        num_heads = 1

    dropout = 0.1
    pooling = "mean"
    max_len = 512
    pad_idx = 0

    return {
        "vocab_size": int(vocab_size),
        "model_dim": int(model_dim),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "ff_mult": int(ff_mult),
        "dropout": float(dropout),
        "pooling": str(pooling),
        "max_len": int(max_len),
        "pad_idx": int(pad_idx),
    }


def _print_multiclass_val_accuracy(tag: str, ckpt_path: str, model_ctor, train_df, test_df):
    if not os.path.exists(ckpt_path):
        print(f"[skip] {tag}: missing {ckpt_path}")
        return

    x_train, x_val, y_train, y_val = get_tfidf(train_df, test_df)

    device = get_device()
    model = model_ctor(n_inputs=x_train.shape[1]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    preds = []
    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(x_val), batch_size):
            end = min(start + batch_size, len(x_val))
            xb = torch.tensor(x_val[start:end], dtype=torch.float32, device=device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds.extend(pred)

    preds = np.array(preds, dtype=np.int64)
    acc = float((preds == y_val).mean())
    print(f"\n[{tag}] CategoryAccuracy = {acc}")
    print(f"N = {len(y_val)}")


def _pairwise_category_accuracy(tag: str, ckpt_path: str, train_df):
    if not os.path.exists(ckpt_path):
        print(f"[skip] {tag}: missing {ckpt_path}")
        return

    train_rows, val_rows = train_test_split(
        train_df,
        test_size=0.2,
        random_state=40,
        stratify=train_df["label"].astype(str),
    )

    def build_pairwise_texts(df):
        texts, gids, opts = [], [], []
        for _, r in df.iterrows():
            s = str(r["FalseSent"])
            gid = r["id"]
            for opt in OPTIONS:
                texts.append(f"{s} [SEP] {str(r[f'Option{opt}'])}")
                gids.append(gid)
                opts.append(opt)
        return texts, np.array(gids), np.array(opts)

    tr_texts, _, _ = build_pairwise_texts(train_rows)
    va_texts, va_gids, va_opts = build_pairwise_texts(val_rows)

    vec = TfidfVectorizer(binary=True, ngram_range=(1, 2))
    vec.fit(tr_texts)

    expected_in = infer_expected_in_dim(ckpt_path)
    got_in = len(vec.vocabulary_)
    if got_in != expected_in:
        raise RuntimeError(
            f"{tag} vocab mismatch: vectorizer={got_in} vs checkpoint expects {expected_in}. "
            f"Make sure training and this use same split/seed."
        )

    device = get_device()
    model = MLP(n_inputs=expected_in, n_hidden=[64], n_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    batch_size = 512
    best = {}

    with torch.no_grad():
        for start in range(0, len(va_texts), batch_size):
            end = min(start + batch_size, len(va_texts))
            X = vec.transform(va_texts[start:end]).toarray().astype(np.float32)
            logits = model(torch.tensor(X, dtype=torch.float32, device=device)).squeeze(1)
            scores = torch.sigmoid(logits).cpu().numpy()

            for s, gid, opt in zip(scores, va_gids[start:end], va_opts[start:end]):
                s = float(s)
                if gid not in best or s > best[gid][0]:
                    best[gid] = (s, opt)

    gold_by_id = val_rows.set_index("id")["label"].to_dict()
    correct = 0
    for gid, (_, pred_opt) in best.items():
        if pred_opt == gold_by_id[gid]:
            correct += 1

    acc = correct / len(val_rows)
    print(f"\n[{tag}] CategoryAccuracy = {acc}")
    print(f"N = {len(val_rows)}")


def _build_pairwise_texts(df: pd.DataFrame):
    texts, gids, opts = [], [], []
    for _, r in df.iterrows():
        s = str(r["FalseSent"])
        gid = r["id"]
        for opt in OPTIONS:
            texts.append(f"{s} [SEP] {str(r[f'Option{opt}'])}")
            gids.append(gid)
            opts.append(opt)
    return texts, np.array(gids), np.array(opts)


def _textresnet_pairwise_category_accuracy(tag: str, ckpt_path: str, train_df, seed=40, max_len=128, min_freq=2, max_vocab=50000):
    if not os.path.exists(ckpt_path):
        print(f"[skip] {tag}: missing {ckpt_path}")
        return

    from src.processing import preprocess_cnn
    from src.models.cnn_text import TextResNetCNN

    X_tr, X_va, y_tr, y_va, vocab, va_group_ids, va_opts, va_gold = preprocess_cnn(
        train_df=train_df,
        seed=seed,
        mode="pairwise",
        max_len=max_len,
        min_freq=min_freq,
        max_vocab=max_vocab,
    )

    device = get_device()
    model = TextResNetCNN(vocab_size=len(vocab), num_classes=1, pad_idx=0).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    batch_size = 256
    scores_list = []
    with torch.no_grad():
        for start in range(0, len(X_va), batch_size):
            end = min(start + batch_size, len(X_va))
            xb = torch.tensor(X_va[start:end], dtype=torch.long, device=device)
            logits = model(xb).squeeze(1)
            scores = torch.sigmoid(logits).cpu().numpy()
            scores_list.append(scores)

    scores = np.concatenate(scores_list, axis=0)

    best = {}
    for s, gid, opt in zip(scores, va_group_ids, va_opts):
        s = float(s)
        if gid not in best or s > best[gid][0]:
            best[gid] = (s, opt)

    correct = 0
    total = 0
    for gid, (_, pred_opt) in best.items():
        total += 1
        if pred_opt == va_gold[gid]:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    print(f"\n[{tag}] CategoryAccuracy = {acc}")
    print(f"N = {total}")



def _load_transformer_vocab_and_meta():
    vocab_path = "checkpoints/transformer_pairwise_vocab.json"
    meta_path  = "checkpoints/transformer_pairwise_meta.json"

    if not os.path.exists(vocab_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            "Missing transformer vocab/meta. Train first so these exist:\n"
            f" - {vocab_path}\n - {meta_path}"
        )

    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return vocab, meta


def _transformer_pairwise_category_accuracy(tag: str, ckpt_path: str, train_df):
    if not os.path.exists(ckpt_path):
        print(f"[skip] {tag}: missing {ckpt_path}")
        return

    from src.models.transformer import TextTransformer
    from src.processing import make_pairwise_rows, encode_text

    vocab, meta = _load_transformer_vocab_and_meta()
    seed    = int(meta["seed"])
    max_len = int(meta["max_len"])

    # Recreate the SAME val split as training
    tr_rows, va_rows = train_test_split(
        train_df,
        test_size=0.2,
        random_state=seed,
        stratify=train_df["label"].astype(str),
    )

    va_pairs = make_pairwise_rows(va_rows)
    X_va = np.array([encode_text(t, vocab, max_len) for t in va_pairs["text"].tolist()], dtype=np.int64)

    va_group_ids = va_pairs["group_id"].values
    va_opts = va_pairs["opt"].values
    va_gold = va_rows.set_index("id")["label"].to_dict()

    hp = infer_transformer_hparams(ckpt_path)
    device = get_device()

    model = TextTransformer(
        vocab_size=len(vocab),
        num_classes=1,
        pad_idx=hp["pad_idx"],
        model_dim=hp["model_dim"],
        num_heads=hp["num_heads"],
        num_layers=hp["num_layers"],
        ff_mult=hp["ff_mult"],
        dropout=hp["dropout"],
        max_len=hp["max_len"],
        pooling=hp["pooling"],
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    batch_size = 256
    scores_chunks = []
    with torch.no_grad():
        for start in range(0, len(X_va), batch_size):
            end = min(start + batch_size, len(X_va))
            xb = torch.tensor(X_va[start:end], dtype=torch.long, device=device)
            logits = model(xb).squeeze(1)
            scores_chunks.append(torch.sigmoid(logits).cpu().numpy())

    scores = np.concatenate(scores_chunks, axis=0)

    best = {}
    for s, gid, opt in zip(scores, va_group_ids, va_opts):
        s = float(s)
        if gid not in best or s > best[gid][0]:
            best[gid] = (s, opt)

    correct = 0
    total = 0
    for gid, (_, pred_opt) in best.items():
        total += 1
        if pred_opt == va_gold[gid]:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    print(f"\n[{tag}] CategoryAccuracy = {acc}")
    print(f"N = {total}")


def _make_submission_transformer_pairwise(train_df, test_df, out_dir: str):
    ckpt_path = "checkpoints/transformer_pairwise.pt"
    if not os.path.exists(ckpt_path):
        print(f"[skip] transformer_pairwise submit: missing {ckpt_path}")
        return

    from src.models.transformer import TextTransformer
    from src.processing import encode_text

    vocab, meta = _load_transformer_vocab_and_meta()
    max_len = int(meta["max_len"])

    # Build test pair texts (same as everywhere else)
    te_texts, te_gids, te_opts = _build_pairwise_texts(test_df)
    X_te = np.array([encode_text(t, vocab, max_len) for t in te_texts], dtype=np.int64)

    hp = infer_transformer_hparams(ckpt_path)
    device = get_device()

    model = TextTransformer(
        vocab_size=len(vocab),
        num_classes=1,
        pad_idx=hp["pad_idx"],
        model_dim=hp["model_dim"],
        num_heads=hp["num_heads"],
        num_layers=hp["num_layers"],
        ff_mult=hp["ff_mult"],
        dropout=hp["dropout"],
        max_len=hp["max_len"],
        pooling=hp["pooling"],
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    batch_size = 256
    best = {}

    with torch.no_grad():
        for start in range(0, len(X_te), batch_size):
            end = min(start + batch_size, len(X_te))
            xb = torch.tensor(X_te[start:end], dtype=torch.long, device=device)
            logits = model(xb).squeeze(1)
            scores = torch.sigmoid(logits).cpu().numpy()

            for s, gid, opt in zip(scores, te_gids[start:end], te_opts[start:end]):
                s = float(s)
                if gid not in best or s > best[gid][0]:
                    best[gid] = (s, opt)

    ids = test_df["id"].tolist()
    answers = [best[i][1] for i in ids]
    out = pd.DataFrame({"id": ids, "answer": answers})
    _save_csv(out, os.path.join(out_dir, "submission_transformer_pairwise.csv"))
def print_all_validation_accuracies(train_df, test_df):
    _print_multiclass_val_accuracy(
        tag="mlp_tfidf",
        ckpt_path="checkpoints/mlp_tfidf.pt",
        model_ctor=lambda n_inputs: MLP(n_inputs=n_inputs, n_hidden=[64], n_classes=3),
        train_df=train_df,
        test_df=test_df,
    )

    _pairwise_category_accuracy(
        tag="mlp_pairwise",
        ckpt_path="checkpoints/mlp_pairwise.pt",
        train_df=train_df,
    )

    _textresnet_pairwise_category_accuracy(
        tag="textresnet_pairwise",
        ckpt_path="checkpoints/cnn_pairwise.pt",
        train_df=train_df,
    )

    _transformer_pairwise_category_accuracy(
        tag="transformer_pairwise",
        ckpt_path="checkpoints/transformer_pairwise.pt",
        train_df=train_df,
    )


def _save_csv(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} rows)")


def _make_submission_mlp_tfidf(train_df, test_df, out_dir: str):
    ckpt_path = "checkpoints/mlp_tfidf.pt"
    if not os.path.exists(ckpt_path):
        print(f"[skip] mlp_tfidf submit: missing {ckpt_path}")
        return

    expected_in = infer_expected_in_dim(ckpt_path)

    train_texts = (
        train_df["FalseSent"].astype(str) + " [SEP] " +
        train_df["OptionA"].astype(str) + " [SEP] " +
        train_df["OptionB"].astype(str) + " [SEP] " +
        train_df["OptionC"].astype(str)
    ).tolist()

    test_texts = (
        test_df["FalseSent"].astype(str) + " [SEP] " +
        test_df["OptionA"].astype(str) + " [SEP] " +
        test_df["OptionB"].astype(str) + " [SEP] " +
        test_df["OptionC"].astype(str)
    ).tolist()

    vec = TfidfVectorizer(binary=True, ngram_range=(1, 2), max_features=expected_in)
    vec.fit(train_texts)

    got_in = len(vec.vocabulary_)
    if got_in != expected_in:
        raise RuntimeError(
            f"mlp_tfidf vocab mismatch: vectorizer={got_in} vs checkpoint expects {expected_in}. "
            f"To fix permanently, save/load vec.vocabulary_ from training."
        )

    device = get_device()
    model = MLP(n_inputs=expected_in, n_hidden=[64], n_classes=3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    batch_size = 256
    preds = []
    with torch.no_grad():
        for start in range(0, len(test_texts), batch_size):
            end = min(start + batch_size, len(test_texts))
            X = vec.transform(test_texts[start:end]).toarray().astype(np.float32)
            logits = model(torch.tensor(X, dtype=torch.float32, device=device))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    answers = [IDX2LBL[int(i)] for i in preds]
    out = pd.DataFrame({"id": test_df["id"].tolist(), "answer": answers})
    _save_csv(out, os.path.join(out_dir, "submission_mlp_tfidf.csv"))


def _make_submission_mlp_pairwise(train_df, test_df, out_dir: str):
    ckpt_path = "checkpoints/mlp_pairwise.pt"
    if not os.path.exists(ckpt_path):
        print(f"[skip] mlp_pairwise submit: missing {ckpt_path}")
        return

    expected_in = infer_expected_in_dim(ckpt_path)

    train_rows, _ = train_test_split(
        train_df,
        test_size=0.2,
        random_state=40,
        stratify=train_df["label"].astype(str),
    )

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

    tr_texts, _, _ = build_pair_texts(train_rows)
    te_texts, te_gids, te_opts = build_pair_texts(test_df)

    vec = TfidfVectorizer(binary=True, ngram_range=(1, 2), max_features=expected_in)
    vec.fit(tr_texts)

    got_in = len(vec.vocabulary_)
    if got_in != expected_in:
        raise RuntimeError(
            f"mlp_pairwise vocab mismatch: vectorizer={got_in} vs checkpoint expects {expected_in}. "
            f"Make sure training and submission use same split/seed=40."
        )

    device = get_device()
    model = MLP(n_inputs=expected_in, n_hidden=[64], n_classes=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    batch_size = 512
    best = {}
    with torch.no_grad():
        for start in range(0, len(te_texts), batch_size):
            end = min(start + batch_size, len(te_texts))
            X = vec.transform(te_texts[start:end]).toarray().astype(np.float32)
            logits = model(torch.tensor(X, dtype=torch.float32, device=device)).squeeze(1)
            scores = torch.sigmoid(logits).cpu().numpy()

            for s, gid, opt in zip(scores, te_gids[start:end], te_opts[start:end]):
                s = float(s)
                if gid not in best or s > best[gid][0]:
                    best[gid] = (s, opt)

    ids = test_df["id"].tolist()
    answers = [best[i][1] for i in ids]
    out = pd.DataFrame({"id": ids, "answer": answers})
    _save_csv(out, os.path.join(out_dir, "submission_mlp_pairwise.csv"))


def _make_submission_textresnet_pairwise(train_df, test_df, out_dir: str, seed=40, max_len=128, min_freq=2, max_vocab=50000):
    ckpt_path = "checkpoints/cnn_pairwise.pt"
    if not os.path.exists(ckpt_path):
        print(f"[skip] textresnet_pairwise submit: missing {ckpt_path}")
        return

    from src.processing import preprocess_cnn, encode_text
    from src.models.cnn_text import TextResNetCNN

    X_tr, X_va, y_tr, y_va, vocab, va_group_ids, va_opts, va_gold = preprocess_cnn(
        train_df=train_df,
        seed=seed,
        mode="pairwise",
        max_len=max_len,
        min_freq=min_freq,
        max_vocab=max_vocab,
    )

    te_texts, te_gids, te_opts = _build_pairwise_texts(test_df)
    X_te = np.array([encode_text(t, vocab, max_len) for t in te_texts], dtype=np.int64)

    device = get_device()
    model = TextResNetCNN(vocab_size=len(vocab), num_classes=1, pad_idx=0).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    batch_size = 256
    best = {}
    with torch.no_grad():
        for start in range(0, len(X_te), batch_size):
            end = min(start + batch_size, len(X_te))
            xb = torch.tensor(X_te[start:end], dtype=torch.long, device=device)
            logits = model(xb).squeeze(1)
            scores = torch.sigmoid(logits).cpu().numpy()

            for s, gid, opt in zip(scores, te_gids[start:end], te_opts[start:end]):
                s = float(s)
                if gid not in best or s > best[gid][0]:
                    best[gid] = (s, opt)

    ids = test_df["id"].tolist()
    answers = [best[i][1] for i in ids]
    out = pd.DataFrame({"id": ids, "answer": answers})
    _save_csv(out, os.path.join(out_dir, "submission_textresnet_pairwise.csv"))


def _make_submission_transformer_pairwise(
    train_df,
    test_df,
    out_dir: str,
    seed: int = 40,
    max_len: int = 128,
    min_freq: int = 2,
    max_vocab: int = 50000,
):
    ckpt_path = "checkpoints/transformer_pairwise.pt"
    if not os.path.exists(ckpt_path):
        print(f"[skip] transformer_pairwise submit: missing {ckpt_path}")
        return

    from src.processing import preprocess_cnn, encode_text
    from src.models.transformer import TextTransformer

    X_tr, X_va, y_tr, y_va, vocab, va_group_ids, va_opts, va_gold = preprocess_cnn(
        train_df=train_df,
        seed=seed,
        mode="pairwise",
        max_len=max_len,
        min_freq=min_freq,
        max_vocab=max_vocab,
    )

    te_texts, te_gids, te_opts = _build_pairwise_texts(test_df)
    X_te = np.array([encode_text(t, vocab, max_len) for t in te_texts], dtype=np.int64)

    hp = infer_transformer_hparams(ckpt_path)

    device = get_device()
    model = TextTransformer(
        vocab_size=len(vocab),
        num_classes=1,
        pad_idx=hp["pad_idx"],
        model_dim=hp["model_dim"],
        num_heads=hp["num_heads"],
        num_layers=hp["num_layers"],
        ff_mult=hp["ff_mult"],
        dropout=hp["dropout"],
        max_len=hp["max_len"],
        pooling=hp["pooling"],
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    batch_size = 256
    best = {}

    try:
        with torch.no_grad():
            for start in range(0, len(X_te), batch_size):
                end = min(start + batch_size, len(X_te))
                xb = torch.tensor(X_te[start:end], dtype=torch.long, device=device)
                logits = model(xb).squeeze(1)
                scores = torch.sigmoid(logits).cpu().numpy()

                for s, gid, opt in zip(scores, te_gids[start:end], te_opts[start:end]):
                    s = float(s)
                    if gid not in best or s > best[gid][0]:
                        best[gid] = (s, opt)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device.type == "mps":
            device = torch.device("cpu")
            model = model.to(device)
            best = {}
            with torch.no_grad():
                for start in range(0, len(X_te), batch_size):
                    end = min(start + batch_size, len(X_te))
                    xb = torch.tensor(X_te[start:end], dtype=torch.long, device=device)
                    logits = model(xb).squeeze(1)
                    scores = torch.sigmoid(logits).cpu().numpy()

                    for s, gid, opt in zip(scores, te_gids[start:end], te_opts[start:end]):
                        s = float(s)
                        if gid not in best or s > best[gid][0]:
                            best[gid] = (s, opt)
        else:
            raise

    ids = test_df["id"].tolist()
    answers = [best[i][1] for i in ids]
    out = pd.DataFrame({"id": ids, "answer": answers})
    _save_csv(out, os.path.join(out_dir, "submission_transformer_pairwise.csv"))


def main():
    x_train_raw = pd.read_csv("data/train_data.csv")
    x_test_raw = pd.read_csv("data/test_data.csv")
    y_train_raw = pd.read_csv("data/train_answers.csv")

    train_df, test_df = preprocess(x_train_raw, x_test_raw, y_train_raw)

    labels = train_df["label"].astype(str).str.strip().values
    print("labels dtype:", labels.dtype, "shape:", labels.shape, "first 5:", labels[:5])

    out_dir = "submits"
    print("Writing submissions to:", os.path.abspath(out_dir))

    _make_submission_mlp_tfidf(train_df, test_df, out_dir)
    _make_submission_mlp_pairwise(train_df, test_df, out_dir)
    _make_submission_textresnet_pairwise(train_df, test_df, out_dir)
    _make_submission_transformer_pairwise(train_df, test_df, out_dir)

    print_all_validation_accuracies(train_df, test_df)


if __name__ == "__main__":
    main()
