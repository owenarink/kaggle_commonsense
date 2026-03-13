from dependencies.dependencies import *
import json, os

IDX2LBL = {0: "A", 1: "B", 2: "C"}
OPTIONS = ["A", "B", "C"]


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_transformer_hparams():
    path = "checkpoints/transformer_pairwise_hparams.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Train transformer first so it gets saved.")
    with open(path, "r") as f:
        return json.load(f)

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
    """
    BBPE + GROUPED evaluation for your grouped-CE transformer.
    This matches training: X shape [N,3,L], labels are {A,B,C} -> {0,1,2},
    pick argmax over the 3 option scores.
    """
    if not os.path.exists(ckpt_path):
        print(f"[skip] {tag}: missing {ckpt_path}")
        return

    from tokenizers import Tokenizer
    from src.models.transformer import TextTransformer

    # Load the exact hparams used in training
    hp = load_transformer_hparams()
    tokenizer_path = hp.get("bbpe_tokenizer_path", "checkpoints/bbpe_tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Missing {tokenizer_path}. Train BBPE transformer first.")

    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = int(tok.get_vocab_size())  # should be 12000

    seed = int(hp.get("seed", 40))
    max_len = int(hp.get("grouped_max_len", 128))

    pad_id = hp.get("pad_idx", None)
    if pad_id is None:
        pad_id = tok.token_to_id("<pad>")
    if pad_id is None:
        pad_id = 0
    pad_id = int(pad_id)

    # Recreate SAME split by question
    tr_rows, va_rows = train_test_split(
        train_df,
        test_size=0.2,
        random_state=seed,
        stratify=train_df["label"].astype(str),
    )

    mapping = {"A": 0, "B": 1, "C": 2}
    y_va = np.array([mapping[x] for x in va_rows["label"].astype(str).str.strip().values], dtype=np.int64)

    # Build grouped X_va: [N,3,L]
    X_va = np.zeros((len(va_rows), 3, max_len), dtype=np.int64)

    for i, r in enumerate(va_rows.itertuples(index=False)):
        fs = str(getattr(r, "FalseSent"))
        oa = str(getattr(r, "OptionA"))
        ob = str(getattr(r, "OptionB"))
        oc = str(getattr(r, "OptionC"))
        texts = [f"{fs} [SEP] {oa}", f"{fs} [SEP] {ob}", f"{fs} [SEP] {oc}"]

        enc = tok.encode_batch(texts)
        for j in range(3):
            ids = enc[j].ids[:max_len]
            if len(ids) < max_len:
                ids = ids + [pad_id] * (max_len - len(ids))
            X_va[i, j] = np.array(ids, dtype=np.int64)

    device = get_device()

    model = TextTransformer(
        vocab_size=vocab_size,               # ✅ 12000
        num_classes=1,                       # scorer per option
        pad_idx=pad_id,
        model_dim=int(hp["model_dim"]),
        num_heads=int(hp["num_heads"]),
        num_layers=int(hp["num_layers"]),
        ff_mult=int(hp["ff_mult"]),
        dropout=float(hp["dropout"]),
        max_len=int(hp["max_len"]),
        pooling=str(hp["pooling"]),
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Batched eval to avoid MPS spikes
    batch_q = 128
    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(X_va), batch_q):
            end = min(start + batch_q, len(X_va))
            xb = torch.tensor(X_va[start:end], dtype=torch.long, device=device)  # [B,3,L]
            yb = torch.tensor(y_va[start:end], dtype=torch.long, device=device)  # [B]

            B, K, L = xb.shape
            scores = model(xb.view(B * K, L)).squeeze(-1).view(B, K)            # [B,3]
            pred = torch.argmax(scores, dim=1)

            correct += (pred == yb).sum().item()
            total += B

            if device.type == "mps":
                torch.mps.empty_cache()

    acc = correct / max(total, 1)
    print(f"\n[{tag}] CategoryAccuracy = {acc}")
    print(f"N = {total}")


def _make_submission_transformer_pairwise(train_df, test_df, out_dir: str):
    ckpt_path = "checkpoints/transformer_pairwise.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/transformer_pairwise_best.pt"
    if not os.path.exists(ckpt_path):
        print("[skip] transformer submit: missing checkpoint")
        return

    from tokenizers import Tokenizer
    from src.models.transformer import TextTransformer

    tokenizer_path = "checkpoints/bbpe_tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Missing {tokenizer_path}. Train BBPE transformer first.")

    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = int(tok.get_vocab_size())

    # make these consistent with training
    max_len = 128  # <-- set to what you used in training for BBPE
    pad_id = tok.token_to_id("<pad>")
    unk_id = tok.token_to_id("<unk>")
    if pad_id is None:
        pad_id = 0
    if unk_id is None:
        unk_id = 1

    # IMPORTANT: build grouped inputs [N,3,L] (same format you trained with grouped CE)
    ids = test_df["id"].astype(str).tolist()

    X_te = np.zeros((len(test_df), 3, max_len), dtype=np.int64)

    # encode each option separately: "FalseSent [SEP] OptionX"
    for i, r in enumerate(test_df.itertuples(index=False)):
        fs = str(getattr(r, "FalseSent"))
        oa = str(getattr(r, "OptionA"))
        ob = str(getattr(r, "OptionB"))
        oc = str(getattr(r, "OptionC"))
        texts = [f"{fs} [SEP] {oa}", f"{fs} [SEP] {ob}", f"{fs} [SEP] {oc}"]

        enc = tok.encode_batch(texts)
        for j in range(3):
            t_ids = enc[j].ids[:max_len]
            if len(t_ids) < max_len:
                t_ids = t_ids + [pad_id] * (max_len - len(t_ids))
            X_te[i, j] = np.array(t_ids, dtype=np.int64)

    # model hyperparams MUST match checkpoint
    hp = infer_transformer_hparams(ckpt_path)
    device = get_device()

    model = TextTransformer(
        vocab_size=vocab_size,            # ✅ 12000
        num_classes=1,                    # scorer per option (grouped CE uses [B,3] scores)
        pad_idx=pad_id,                   # ✅ pad id from tokenizer
        model_dim=hp["model_dim"],
        num_heads=hp["num_heads"],
        num_layers=hp["num_layers"],
        ff_mult=hp["ff_mult"],
        dropout=hp["dropout"],
        max_len=max(hp["max_len"], max_len),
        pooling=hp["pooling"],
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # predict in batches to avoid MPS OOM
    batch_q = 128  # questions per batch (=> 384 sequences)
    preds = []

    with torch.no_grad():
        for start in range(0, len(X_te), batch_q):
            end = min(start + batch_q, len(X_te))
            xb = torch.tensor(X_te[start:end], dtype=torch.long, device=device)  # [B,3,L]
            B, K, L = xb.shape
            scores = model(xb.view(B * K, L)).view(B, K)                         # [B,3]
            pred = torch.argmax(scores, dim=1).cpu().numpy().tolist()
            preds.extend(pred)

            if device.type == "mps":
                torch.mps.empty_cache()

    answers = [OPTIONS[i] for i in preds]
    out = pd.DataFrame({"id": ids, "answer": answers})

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission_transformer_pairwise.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out)} rows)")

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
        ckpt_path=("checkpoints/transformer_pairwise_best.pt"
                   if os.path.exists("checkpoints/transformer_pairwise_best.pt")
                   else "checkpoints/transformer_pairwise.pt"),
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


def _make_submission_transformer_pairwise_UNUSED(
    train_df,
    test_df,
    out_dir: str,
    seed: int = 40,
    max_len: int = 128,
    min_freq: int = 2,
    max_vocab: int = 50000,
):
    
    ckpt_path = "checkpoints/transformer_pairwise_best.pt"  # <- use BEST
    if not os.path.exists(ckpt_path):
        # fallback if you didn't rename yet
        ckpt_path = "checkpoints/transformer_pairwise.pt"
    if not os.path.exists(ckpt_path):
        print(f"[skip] transformer_pairwise submit: missing checkpoint")
        return

    from src.models.transformer import TextTransformer
    from src.processing import encode_text

    # IMPORTANT: reuse training vocab/meta (do NOT call preprocess_cnn here)
    vocab, meta = _load_transformer_vocab_and_meta()
    max_len = int(meta["max_len"])

    # Build test pair texts and encode with SAME vocab
    te_texts, te_gids, te_opts = _build_pairwise_texts(test_df)
    X_te = np.array([encode_text(t, vocab, max_len) for t in te_texts], dtype=np.int64)


    hp = load_transformer_hparams()
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

    def _run_on_device(dev):
        nonlocal best
        best = {}
        with torch.no_grad():
            for start in range(0, len(X_te), batch_size):
                end = min(start + batch_size, len(X_te))
                xb = torch.tensor(X_te[start:end], dtype=torch.long, device=dev)
                logits = model(xb).squeeze(1)
                scores = torch.sigmoid(logits).detach().cpu().numpy()
                for s, gid, opt in zip(scores, te_gids[start:end], te_opts[start:end]):
                    s = float(s)
                    if gid not in best or s > best[gid][0]:
                        best[gid] = (s, opt)

    try:
        _run_on_device(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device.type == "mps":
            device = torch.device("cpu")
            model = model.to(device)
            _run_on_device(device)
        else:
            raise

    ids = test_df["id"].tolist()
    answers = [best[i][1] for i in ids]
    out = pd.DataFrame({"id": ids, "answer": answers})

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submission_transformer_pairwise.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out)} rows)") 

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
