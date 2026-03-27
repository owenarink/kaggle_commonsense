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


def load_hparams(path: str, fallback_ckpt: str = None):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    if fallback_ckpt is not None:
        return infer_transformer_hparams(fallback_ckpt)
    raise FileNotFoundError(f"Missing hparams file: {path}")


def first_existing_file(paths):
    return first_existing(paths)


def first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def checkpoint_kind(ckpt_path: str) -> str:
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj.get("model_state", obj)
    keys = list(sd.keys())
    if any("latent_edit." in k for k in keys):
        return "lec"
    if any("cross_option." in k for k in keys):
        return "cross_option"
    if any("counterfact." in k for k in keys):
        return "counterfact"
    if any(".self_attn.qkv_proj." in k for k in keys):
        return "plain_transformer"
    if any(".self_attn.q_proj." in k for k in keys):
        return "deberta_like"
    return "unknown"


def first_existing_matching(paths, expected_kind: str):
    for path in paths:
        if path and os.path.exists(path) and checkpoint_kind(path) == expected_kind:
            return path
    return None

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

def _build_grouped_bbpe_inputs(df: pd.DataFrame, hp: dict):
    from src.tokenizer_bbpe import load_bbpe, encode_grouped_bbpe

    tokenizer_path = hp.get("tokenizer_path") or hp.get("bbpe_tokenizer_path") or "checkpoints/bbpe_tokenizer.json"
    meta_path = hp.get("tokenizer_meta_path")
    if not meta_path and tokenizer_path.endswith("_tokenizer.json"):
        meta_path = tokenizer_path.replace("_tokenizer.json", "_tokenizer_meta.json")
    if not meta_path:
        meta_path = "checkpoints/bbpe_tokenizer_meta.json"
    tok, meta = load_bbpe(tokenizer_path, meta_path)
    pad_id = int(hp.get("pad_idx", meta.get("pad_id", 0)))
    max_len = int(hp.get("grouped_max_len", 128))
    X = encode_grouped_bbpe(df, tok, max_len=max_len, pad_id=pad_id)
    return X, meta


def _grouped_model_scores(model, xb: torch.Tensor, grouped_forward: bool):
    if grouped_forward:
        return model(xb)
    B, K, L = xb.shape
    return model(xb.view(B * K, L)).squeeze(-1).view(B, K)


def _safe_load_grouped_state(model, ckpt_path: str, device, tag: str) -> bool:
    state = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state)
        return True
    except RuntimeError as e:
        print(f"[skip] {tag}: checkpoint/model mismatch for {ckpt_path}")
        print(f"reason: {e}")
        return False


def _grouped_bbpe_category_accuracy(tag: str, ckpt_path: str, hparams_path: str, model_builder, train_df, grouped_forward: bool):
    if not os.path.exists(ckpt_path):
        print(f"[skip] {tag}: missing {ckpt_path}")
        return

    hp = load_hparams(hparams_path, fallback_ckpt=ckpt_path)
    seed = int(hp.get("seed", 40))
    tr_rows, va_rows = train_test_split(
        train_df,
        test_size=0.2,
        random_state=seed,
        stratify=train_df["label"].astype(str),
    )

    mapping = {"A": 0, "B": 1, "C": 2}
    y_va = np.array([mapping[x] for x in va_rows["label"].astype(str).str.strip().values], dtype=np.int64)
    X_va, meta = _build_grouped_bbpe_inputs(va_rows, hp)

    device = get_device()
    model = model_builder(hp, meta).to(device)
    if not _safe_load_grouped_state(model, ckpt_path, device, tag):
        return
    model.eval()

    batch_q = 128
    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(X_va), batch_q):
            end = min(start + batch_q, len(X_va))
            xb = torch.tensor(X_va[start:end], dtype=torch.long, device=device)  # [B,3,L]
            yb = torch.tensor(y_va[start:end], dtype=torch.long, device=device)  # [B]
            scores = _grouped_model_scores(model, xb, grouped_forward=grouped_forward)
            pred = torch.argmax(scores, dim=1)

            correct += (pred == yb).sum().item()
            total += xb.size(0)

            if device.type == "mps":
                torch.mps.empty_cache()

    acc = correct / max(total, 1)
    print(f"\n[{tag}] CategoryAccuracy = {acc}")
    print(f"N = {total}")
    if tag == "transformer_bbpe" and acc < 0.40:
        print(
            f"[warn] {tag}: very low validation accuracy usually means the BBPE tokenizer/checkpoint pair no longer matches "
            f"(for example, the shared tokenizer file was retrained after this checkpoint was created)."
        )


def _make_submission_grouped_bbpe(tag: str, ckpt_path: str, hparams_path: str, model_builder, test_df, out_dir: str, out_name: str, grouped_forward: bool):
    if not os.path.exists(ckpt_path):
        print(f"[skip] {tag} submit: missing {ckpt_path}")
        return

    hp = load_hparams(hparams_path, fallback_ckpt=ckpt_path)
    X_te, meta = _build_grouped_bbpe_inputs(test_df, hp)

    device = get_device()
    model = model_builder(hp, meta).to(device)
    if not _safe_load_grouped_state(model, ckpt_path, device, tag):
        return
    model.eval()

    batch_q = 128
    preds = []

    with torch.no_grad():
        for start in range(0, len(X_te), batch_q):
            end = min(start + batch_q, len(X_te))
            xb = torch.tensor(X_te[start:end], dtype=torch.long, device=device)
            scores = _grouped_model_scores(model, xb, grouped_forward=grouped_forward)
            preds.extend(torch.argmax(scores, dim=1).cpu().numpy().tolist())

            if device.type == "mps":
                torch.mps.empty_cache()

    answers = [OPTIONS[i] for i in preds]
    out = pd.DataFrame({"id": test_df["id"].astype(str).tolist(), "answer": answers})
    _save_csv(out, os.path.join(out_dir, out_name))


def _make_submission_transformer_bbpe(train_df, test_df, out_dir: str):
    from src.models.transformer import TextTransformer

    ckpt_path = first_existing_matching([
        "checkpoints/transformer_best_acc.pt",
        "checkpoints/transformer_best_loss.pt",
        "checkpoints/transformer_pairwise_best.pt",
        "checkpoints/transformer_pairwise.pt",
    ], "plain_transformer")
    if ckpt_path is None:
        print("[skip] transformer_bbpe submit: missing plain-transformer checkpoint")
        return

    def build_model(hp, meta):
        return TextTransformer(
            vocab_size=int(meta["vocab_size"]),
            num_classes=1,
            pad_idx=int(hp.get("pad_idx", meta.get("pad_id", 0))),
            model_dim=int(hp["model_dim"]),
            num_heads=int(hp["num_heads"]),
            num_layers=int(hp["num_layers"]),
            ff_mult=int(hp["ff_mult"]),
            dropout=float(hp["dropout"]),
            max_len=int(hp["max_len"]),
            pooling=str(hp["pooling"]),
        )

    _make_submission_grouped_bbpe(
        tag="transformer_bbpe",
        ckpt_path=ckpt_path,
        hparams_path=first_existing_file([
            "checkpoints/transformer_bbpe_hparams.json",
            "checkpoints/transformer_pairwise_hparams.json",
        ]),
        model_builder=build_model,
        test_df=test_df,
        out_dir=out_dir,
        out_name="submission_transformer_bbpe.csv",
        grouped_forward=False,
    )

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

    from src.models.transformer import TextTransformer
    from src.models.bert import BertTransformer
    from src.models.bert_attention_experiments import BertSepDistanceTransformer, BertRelSimpleTransformer
    from src.models.bertcounterfact import BertCounterFactTransformer
    from src.models.bertcounterfact_cross_option_competition import BertCounterFactCrossOpitionCompetitionTransformer
    from src.models.bertcounter_latent_edit_competition import BertCounterFactLatentEditCompetitionTransformer

    def build_transformer(hp, meta):
        return TextTransformer(
            vocab_size=int(meta["vocab_size"]),
            num_classes=1,
            pad_idx=int(hp.get("pad_idx", meta.get("pad_id", 0))),
            model_dim=int(hp["model_dim"]),
            num_heads=int(hp["num_heads"]),
            num_layers=int(hp["num_layers"]),
            ff_mult=int(hp["ff_mult"]),
            dropout=float(hp["dropout"]),
            max_len=int(hp["max_len"]),
            pooling=str(hp["pooling"]),
        )

    def build_bert(hp, meta):
        return BertTransformer(
            vocab_size=int(meta["vocab_size"]),
            num_classes=1,
            pad_idx=int(hp.get("pad_idx", meta.get("pad_id", 0))),
            model_dim=int(hp["model_dim"]),
            num_heads=int(hp["num_heads"]),
            num_layers=int(hp["num_layers"]),
            ff_mult=int(hp["ff_mult"]),
            dropout=float(hp["dropout"]),
            max_len=int(hp["max_len"]),
            pooling=str(hp["pooling"]),
        )

    def build_bert_sepdist(hp, meta):
        return BertSepDistanceTransformer(
            vocab_size=int(meta["vocab_size"]),
            num_classes=1,
            pad_idx=int(hp.get("pad_idx", meta.get("pad_id", 0))),
            sep_idx=int(hp.get("sep_idx", meta.get("sep_id", 2))),
            model_dim=int(hp["model_dim"]),
            num_heads=int(hp["num_heads"]),
            num_layers=int(hp["num_layers"]),
            ff_mult=int(hp["ff_mult"]),
            dropout=float(hp["dropout"]),
            max_len=int(hp["max_len"]),
            pooling=str(hp["pooling"]),
        )

    def build_bert_relsimple(hp, meta):
        return BertRelSimpleTransformer(
            vocab_size=int(meta["vocab_size"]),
            num_classes=1,
            pad_idx=int(hp.get("pad_idx", meta.get("pad_id", 0))),
            sep_idx=int(hp.get("sep_idx", meta.get("sep_id", 2))),
            model_dim=int(hp["model_dim"]),
            num_heads=int(hp["num_heads"]),
            num_layers=int(hp["num_layers"]),
            ff_mult=int(hp["ff_mult"]),
            dropout=float(hp["dropout"]),
            max_len=int(hp["max_len"]),
            pooling=str(hp["pooling"]),
        )

    def build_counterfact(hp, meta):
        kwargs = dict(
            vocab_size=int(meta["vocab_size"]),
            num_classes=1,
            pad_idx=int(hp.get("pad_idx", meta.get("pad_id", 0))),
            model_dim=int(hp["model_dim"]),
            num_heads=int(hp["num_heads"]),
            num_layers=int(hp["num_layers"]),
            ff_mult=int(hp["ff_mult"]),
            dropout=float(hp["dropout"]),
            max_len=int(hp["max_len"]),
            pooling=str(hp["pooling"]),
        )
        if "sep_idx" in hp:
            kwargs["sep_idx"] = int(hp["sep_idx"])
        return BertCounterFactTransformer(**kwargs)

    def build_cross_option(hp, meta):
        kwargs = dict(
            vocab_size=int(meta["vocab_size"]),
            pad_idx=int(hp.get("pad_idx", meta.get("pad_id", 0))),
            model_dim=int(hp["model_dim"]),
            num_heads=int(hp["num_heads"]),
            num_layers=int(hp["num_layers"]),
            ff_mult=int(hp["ff_mult"]),
            dropout=float(hp["dropout"]),
            max_len=int(hp["max_len"]),
            pooling=str(hp["pooling"]),
        )
        if "sep_idx" in hp:
            kwargs["sep_idx"] = int(hp["sep_idx"])
        return BertCounterFactCrossOpitionCompetitionTransformer(**kwargs)

    def build_lec(hp, meta):
        return BertCounterFactLatentEditCompetitionTransformer(
            vocab_size=int(meta["vocab_size"]),
            pad_idx=int(hp.get("pad_idx", meta.get("pad_id", 0))),
            sep_idx=int(hp.get("sep_idx", meta.get("sep_id", 2))),
            model_dim=int(hp["model_dim"]),
            num_heads=int(hp["num_heads"]),
            num_layers=int(hp["num_layers"]),
            ff_mult=int(hp["ff_mult"]),
            dropout=float(hp["dropout"]),
            max_len=int(hp["max_len"]),
            pooling=str(hp["pooling"]),
            edit_minimality_weight=float(hp.get("edit_minimality_weight", 0.10)),
        )

    grouped_runs = [
        ("transformer_bbpe", first_existing_matching(["checkpoints/transformer_best_acc.pt", "checkpoints/transformer_best_loss.pt", "checkpoints/transformer_pairwise_best.pt", "checkpoints/transformer_pairwise.pt"], "plain_transformer"), first_existing_file(["checkpoints/transformer_bbpe_hparams.json", "checkpoints/transformer_pairwise_hparams.json"]), build_transformer, False),
        ("transformer_attentiontypes", first_existing_matching(["checkpoints/transformer_attentiontypes_best_acc.pt", "checkpoints/transformer_attentiontypes.pt", "checkpoints/transformer_attentiontypes_best_loss.pt", "checkpoints/transformer_best_acc.pt", "checkpoints/transformer_best_loss.pt"], "deberta_like"), first_existing_file(["checkpoints/transformer_attentiontypes_hparams.json", "checkpoints/transformer_pairwise_hparams.json"]), build_bert, False),
        ("transformer_attentiontypes_2", first_existing(["checkpoints/transformer_attentiontypes_2_best_acc.pt", "checkpoints/transformer_attentiontypes_2.pt", "checkpoints/transformer_attentiontypes_2_best_loss.pt"]), first_existing_file(["checkpoints/transformer_attentiontypes_2_hparams.json"]), build_bert, False),
        ("transformer_attention_sepdist", first_existing(["checkpoints/transformer_attention_sepdist_best_acc.pt", "checkpoints/transformer_attention_sepdist.pt", "checkpoints/transformer_attention_sepdist_best_loss.pt"]), first_existing_file(["checkpoints/transformer_attention_sepdist_hparams.json"]), build_bert_sepdist, False),
        ("transformer_attention_relsimple", first_existing(["checkpoints/transformer_attention_relsimple_best_acc.pt", "checkpoints/transformer_attention_relsimple.pt", "checkpoints/transformer_attention_relsimple_best_loss.pt"]), first_existing_file(["checkpoints/transformer_attention_relsimple_hparams.json"]), build_bert_relsimple, False),
        ("transformer_bertcounterfact", first_existing(["checkpoints/transformer_bertcounterfact_best_acc.pt", "checkpoints/transformer_bertcounterfact.pt", "checkpoints/transformer_bertcounterfact_best_loss.pt"]), first_existing_file(["checkpoints/transformer_bertcounterfact_hparams.json", "checkpoints/transformer_pairwise_hparams.json"]), build_counterfact, False),
        ("transformer_bertcounter_cross_option", first_existing(["checkpoints/transformer_bertcounter_cross_option_best_acc.pt", "checkpoints/transformer_bertcounter_cross_option.pt", "checkpoints/transformer_bertcounter_cross_option_best_loss.pt"]), first_existing_file(["checkpoints/transformer_bertcounter_cross_option_hparams.json"]), build_cross_option, True),
        ("transformer_bertcounter_lec", first_existing(["checkpoints/transformer_bertcounter_LEC_best_acc.pt", "checkpoints/transformer_bertcounter_LEC.pt", "checkpoints/transformer_bertcounter_LEC_best_loss.pt"]), first_existing_file(["checkpoints/transformer_bertcounter_LEC_hparams.json"]), build_lec, True),
    ]

    for tag, ckpt_path, hparams_path, model_builder, grouped_forward in grouped_runs:
        if ckpt_path is None:
            print(f"[skip] {tag}: missing checkpoint")
            continue
        _grouped_bbpe_category_accuracy(
            tag=tag,
            ckpt_path=ckpt_path,
            hparams_path=hparams_path,
            model_builder=model_builder,
            train_df=train_df,
            grouped_forward=grouped_forward,
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
    _make_submission_transformer_bbpe(train_df, test_df, out_dir)

    from src.models.bert import BertTransformer
    from src.models.bert_attention_experiments import BertSepDistanceTransformer, BertRelSimpleTransformer
    from src.models.bertcounterfact import BertCounterFactTransformer
    from src.models.bertcounterfact_cross_option_competition import BertCounterFactCrossOpitionCompetitionTransformer
    from src.models.bertcounter_latent_edit_competition import BertCounterFactLatentEditCompetitionTransformer

    def build_bert(hp, meta):
        return BertTransformer(
            vocab_size=int(meta["vocab_size"]),
            num_classes=1,
            pad_idx=int(hp.get("pad_idx", meta.get("pad_id", 0))),
            model_dim=int(hp["model_dim"]),
            num_heads=int(hp["num_heads"]),
            num_layers=int(hp["num_layers"]),
            ff_mult=int(hp["ff_mult"]),
            dropout=float(hp["dropout"]),
            max_len=int(hp["max_len"]),
            pooling=str(hp["pooling"]),
        )

    def build_bert_sepdist(hp, meta):
        return BertSepDistanceTransformer(
            vocab_size=int(meta["vocab_size"]),
            num_classes=1,
            pad_idx=int(hp.get("pad_idx", meta.get("pad_id", 0))),
            sep_idx=int(hp.get("sep_idx", meta.get("sep_id", 2))),
            model_dim=int(hp["model_dim"]),
            num_heads=int(hp["num_heads"]),
            num_layers=int(hp["num_layers"]),
            ff_mult=int(hp["ff_mult"]),
            dropout=float(hp["dropout"]),
            max_len=int(hp["max_len"]),
            pooling=str(hp["pooling"]),
        )

    def build_bert_relsimple(hp, meta):
        return BertRelSimpleTransformer(
            vocab_size=int(meta["vocab_size"]),
            num_classes=1,
            pad_idx=int(hp.get("pad_idx", meta.get("pad_id", 0))),
            sep_idx=int(hp.get("sep_idx", meta.get("sep_id", 2))),
            model_dim=int(hp["model_dim"]),
            num_heads=int(hp["num_heads"]),
            num_layers=int(hp["num_layers"]),
            ff_mult=int(hp["ff_mult"]),
            dropout=float(hp["dropout"]),
            max_len=int(hp["max_len"]),
            pooling=str(hp["pooling"]),
        )

    def build_counterfact(hp, meta):
        kwargs = dict(
            vocab_size=int(meta["vocab_size"]),
            num_classes=1,
            pad_idx=int(hp.get("pad_idx", meta.get("pad_id", 0))),
            model_dim=int(hp["model_dim"]),
            num_heads=int(hp["num_heads"]),
            num_layers=int(hp["num_layers"]),
            ff_mult=int(hp["ff_mult"]),
            dropout=float(hp["dropout"]),
            max_len=int(hp["max_len"]),
            pooling=str(hp["pooling"]),
        )
        if "sep_idx" in hp:
            kwargs["sep_idx"] = int(hp["sep_idx"])
        return BertCounterFactTransformer(**kwargs)

    def build_cross_option(hp, meta):
        kwargs = dict(
            vocab_size=int(meta["vocab_size"]),
            pad_idx=int(hp.get("pad_idx", meta.get("pad_id", 0))),
            model_dim=int(hp["model_dim"]),
            num_heads=int(hp["num_heads"]),
            num_layers=int(hp["num_layers"]),
            ff_mult=int(hp["ff_mult"]),
            dropout=float(hp["dropout"]),
            max_len=int(hp["max_len"]),
            pooling=str(hp["pooling"]),
        )
        if "sep_idx" in hp:
            kwargs["sep_idx"] = int(hp["sep_idx"])
        return BertCounterFactCrossOpitionCompetitionTransformer(**kwargs)

    def build_lec(hp, meta):
        return BertCounterFactLatentEditCompetitionTransformer(
            vocab_size=int(meta["vocab_size"]),
            pad_idx=int(hp.get("pad_idx", meta.get("pad_id", 0))),
            sep_idx=int(hp.get("sep_idx", meta.get("sep_id", 2))),
            model_dim=int(hp["model_dim"]),
            num_heads=int(hp["num_heads"]),
            num_layers=int(hp["num_layers"]),
            ff_mult=int(hp["ff_mult"]),
            dropout=float(hp["dropout"]),
            max_len=int(hp["max_len"]),
            pooling=str(hp["pooling"]),
            edit_minimality_weight=float(hp.get("edit_minimality_weight", 0.10)),
        )

    grouped_submissions = [
        ("transformer_attentiontypes", first_existing_matching(["checkpoints/transformer_attentiontypes_best_acc.pt", "checkpoints/transformer_attentiontypes.pt", "checkpoints/transformer_attentiontypes_best_loss.pt", "checkpoints/transformer_best_acc.pt", "checkpoints/transformer_best_loss.pt"], "deberta_like"), first_existing_file(["checkpoints/transformer_attentiontypes_hparams.json", "checkpoints/transformer_pairwise_hparams.json"]), build_bert, "submission_transformer_attentiontypes.csv", False),
        ("transformer_attentiontypes_2", first_existing(["checkpoints/transformer_attentiontypes_2_best_acc.pt", "checkpoints/transformer_attentiontypes_2.pt", "checkpoints/transformer_attentiontypes_2_best_loss.pt"]), first_existing_file(["checkpoints/transformer_attentiontypes_2_hparams.json"]), build_bert, "submission_transformer_attentiontypes_2.csv", False),
        ("transformer_attention_sepdist", first_existing(["checkpoints/transformer_attention_sepdist_best_acc.pt", "checkpoints/transformer_attention_sepdist.pt", "checkpoints/transformer_attention_sepdist_best_loss.pt"]), first_existing_file(["checkpoints/transformer_attention_sepdist_hparams.json"]), build_bert_sepdist, "submission_transformer_attention_sepdist.csv", False),
        ("transformer_attention_relsimple", first_existing(["checkpoints/transformer_attention_relsimple_best_acc.pt", "checkpoints/transformer_attention_relsimple.pt", "checkpoints/transformer_attention_relsimple_best_loss.pt"]), first_existing_file(["checkpoints/transformer_attention_relsimple_hparams.json"]), build_bert_relsimple, "submission_transformer_attention_relsimple.csv", False),
        ("transformer_bertcounterfact", first_existing(["checkpoints/transformer_bertcounterfact_best_acc.pt", "checkpoints/transformer_bertcounterfact.pt", "checkpoints/transformer_bertcounterfact_best_loss.pt"]), first_existing_file(["checkpoints/transformer_bertcounterfact_hparams.json", "checkpoints/transformer_pairwise_hparams.json"]), build_counterfact, "submission_transformer_bertcounterfact.csv", False),
        ("transformer_bertcounter_cross_option", first_existing(["checkpoints/transformer_bertcounter_cross_option_best_acc.pt", "checkpoints/transformer_bertcounter_cross_option.pt", "checkpoints/transformer_bertcounter_cross_option_best_loss.pt"]), first_existing_file(["checkpoints/transformer_bertcounter_cross_option_hparams.json"]), build_cross_option, "submission_transformer_bertcounter_cross_option.csv", True),
        ("transformer_bertcounter_lec", first_existing(["checkpoints/transformer_bertcounter_LEC_best_acc.pt", "checkpoints/transformer_bertcounter_LEC.pt", "checkpoints/transformer_bertcounter_LEC_best_loss.pt"]), first_existing_file(["checkpoints/transformer_bertcounter_LEC_hparams.json"]), build_lec, "submission_transformer_bertcounter_lec.csv", True),
    ]

    for tag, ckpt_path, hparams_path, model_builder, out_name, grouped_forward in grouped_submissions:
        if ckpt_path is None:
            print(f"[skip] {tag} submit: missing checkpoint")
            continue
        _make_submission_grouped_bbpe(
            tag=tag,
            ckpt_path=ckpt_path,
            hparams_path=hparams_path,
            model_builder=model_builder,
            test_df=test_df,
            out_dir=out_dir,
            out_name=out_name,
            grouped_forward=grouped_forward,
        )

    print_all_validation_accuracies(train_df, test_df)


if __name__ == "__main__":
    main()
