import os, json
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

DEFAULT_TOK_PATH = "checkpoints/bbpe_tokenizer.json"
DEFAULT_META_PATH = "checkpoints/bbpe_tokenizer_meta.json"

def train_bbpe_tokenizer_from_train_df(
    train_df: pd.DataFrame,
    out_path: str = DEFAULT_TOK_PATH,
    meta_path: str = DEFAULT_META_PATH,
    vocab_size: int = 12000,
    min_freq: int = 2,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    # Train on PAIRS, but as plain text is fine
    texts = []
    for _, r in train_df.iterrows():
        s = str(r["FalseSent"])
        for opt in ["A", "B", "C"]:
            texts.append(f"{s} {str(r[f'Option{opt}'])}")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["<pad>", "<unk>", "<sep>", "<cls>", "<eos>"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Force a clean structure: <cls> A <sep> B <eos>
    cls_id = tokenizer.token_to_id("<cls>")
    sep_id = tokenizer.token_to_id("<sep>")
    eos_id = tokenizer.token_to_id("<eos>")

    tokenizer.post_processor = TemplateProcessing(
        single="<cls> $A <eos>",
        pair="<cls> $A <sep> $B <eos>",
        special_tokens=[("<cls>", cls_id), ("<sep>", sep_id), ("<eos>", eos_id)],
    )

    tokenizer.save(out_path)

    meta = {
        "vocab_size": vocab_size,
        "min_freq": min_freq,
        "special_tokens": ["<pad>", "<unk>", "<sep>", "<cls>", "<eos>"],
        "pad_id": tokenizer.token_to_id("<pad>"),
        "unk_id": tokenizer.token_to_id("<unk>"),
        "cls_id": cls_id,
        "sep_id": sep_id,
        "eos_id": eos_id,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print("Saved BBPE tokenizer:", out_path)
    print("Vocab size:", tokenizer.get_vocab_size())

def load_bbpe(tokenizer_path: str = DEFAULT_TOK_PATH, meta_path: str = DEFAULT_META_PATH):
    tok = Tokenizer.from_file(tokenizer_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return tok, meta

def ensure_bbpe(
    train_df: pd.DataFrame,
    tokenizer_path: str = DEFAULT_TOK_PATH,
    meta_path: str = DEFAULT_META_PATH,
    vocab_size: int = 12000,
    min_freq: int = 2,
):
    # retrain if tokenizer OR meta is missing
    if (not os.path.exists(tokenizer_path)) or (not os.path.exists(meta_path)):
        train_bbpe_tokenizer_from_train_df(
            train_df,
            out_path=tokenizer_path,
            meta_path=meta_path,
            vocab_size=vocab_size,
            min_freq=min_freq,
        )

def encode_grouped_bbpe(df: pd.DataFrame, tok: Tokenizer, max_len: int, pad_id: int):
    """
    Returns X: [N,3,L] int64
    Uses pair encoding (FalseSent, OptionX) and TemplateProcessing inserts <cls>, <sep>, <eos>.
    """
    # Ensure consistent trunc/pad
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(length=max_len, pad_id=pad_id, pad_token="<pad>")

    pairs = []
    for _, r in df.iterrows():
        s = str(r["FalseSent"])
        pairs.append((s, str(r["OptionA"])))
        pairs.append((s, str(r["OptionB"])))
        pairs.append((s, str(r["OptionC"])))

    enc = tok.encode_batch(pairs)

    N = len(df)
    X = np.full((N, 3, max_len), pad_id, dtype=np.int64)
    for i, e in enumerate(enc):
        X[i // 3, i % 3, :] = np.asarray(e.ids, dtype=np.int64)

    return X
