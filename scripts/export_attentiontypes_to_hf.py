import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hf_attentiontypes import AttentionTypesConfig, AttentionTypesForSequenceClassification


HF_CODE_DIR = ROOT / "hf_attentiontypes"
DATA_DIR = ROOT / "data"
CHECKPOINTS_DIR = ROOT / "checkpoints"
DEFAULT_HPARAMS = CHECKPOINTS_DIR / "transformer_attentiontypes_hparams.json"
DEFAULT_STATE_DICT = CHECKPOINTS_DIR / "transformer_attentiontypes_best_acc.pt"
DEFAULT_TOKENIZER = CHECKPOINTS_DIR / "transformer_attentiontypes_tokenizer.json"
DEFAULT_TOKENIZER_META = CHECKPOINTS_DIR / "transformer_attentiontypes_tokenizer_meta.json"
DATASET_FILES = [
    "train_data.csv",
    "train_answers.csv",
    "test_data.csv",
    "sample.csv",
]
DEFAULT_DATASET_REPO_ID = "owenarink/kaggle-commonsense"


MODEL_CARD_TEMPLATE = """---
language:
- en
library_name: transformers
pipeline_tag: text-classification
tags:
- custom-code
- transformers
- pytorch
- commonsense-reasoning
- course-project
- experimental
datasets:
- {dataset_repo_id}
---

# {repo_name}

This model is a custom `transformers` export of the `AttentionTypes` model from a course project on commonsense reasoning. The task is to choose the correct answer (`A`, `B`, or `C`) for a false sentence by comparing candidate repairs and selecting the most sensible one.

The underlying model is a grouped BBPE transformer with DeBERTa-inspired disentangled attention. It was trained as a multiple-choice scoring model for a Kaggle-style commonsense challenge and exported here as a custom Hugging Face model for inference and reproducibility. This page corresponds to the regular `AttentionTypes` model from the repository, not the separate hinge-loss variant.

## Model Description

- Architecture: encoder-only transformer with disentangled attention
- Tokenization: grouped BBPE with `<cls> false sentence <sep> option <eos>`
- Objective: score each candidate option and select the highest-scoring repair
- Output: one logit per encoded `(false sentence, option)` pair
- Status: experimental research artifact from an in-progress student project

This is not a pretrained general-purpose language model. It is a task-specific classifier checkpoint.

## Task Summary

One example question looks like this:

- False sentence: `The sun rises in the west.`
- A: `The sun rises in the east.`
- B: `The sun sets in the west.`
- C: `The sun shines at night.`

The correct answer is `A`. The model learns to score which candidate best repairs the false statement.

## Training Data

The model was trained for a course competition setup built around false sentences paired with three candidate corrections. Each training example is represented as three grouped inputs, one per answer option, and the final prediction is the argmax over the three option scores.

The data used for this project lives in the original repository as local CSV files rather than as a standalone Hugging Face dataset:

- `data/train_data.csv`
- `data/train_answers.csv`
- `data/test_data.csv`
- `data/sample.csv`

The training table contains a false sentence plus three candidate answer options. The labels indicate which option correctly repairs the false statement. The test split follows the same structure without released labels.

Because this dataset comes from a course competition setup, it should be treated as task-specific experimental data rather than a broadly curated benchmark.

The same CSV files are included in this Hugging Face repository under `dataset/` for convenience.

## Performance

In the project repository, this `AttentionTypes` variant was the strongest final experiment.

- Best validation accuracy: `0.7456`
- Kaggle public score: `0.7350`

These numbers come from the course-project evaluation setting and should not be treated as a broad benchmark outside that task.

## Architecture

This model uses a shared encoder and linear scoring head for all answer options. For each question, the three candidate options are encoded separately, scored independently, and ranked.

The attention mechanism is a simplified DeBERTa-style disentangled attention with three terms:

- content-to-content
- content-to-position
- position-to-content

Relative position embeddings are used inside attention, and the pooled sequence representation is produced with mean pooling over non-padding tokens.

## Training Setup

- Vocabulary size: `{vocab_size}`
- Model dim: `{model_dim}`
- Heads: `{num_heads}`
- Layers: `{num_layers}`
- FF multiplier: `{ff_mult}`
- Dropout: `{dropout}`
- Pooling: `{pooling}`
- Max relative positions: `{max_position_embeddings}`
- Grouped input max length: `{grouped_max_len}`
- BBPE min frequency: `{bbpe_min_freq}`

## Intended Use

This model is intended for:

- reproducing the course-project experiment
- inspecting a custom disentangled-attention classifier
- running inference on data shaped like the original false-sentence repair task

This model is not intended for:

- general-purpose text classification
- masked language modeling
- use as a drop-in replacement for standard BERT checkpoints

## How To Use

For one question, encode the three `(FalseSent, OptionX)` pairs independently and choose the highest-scoring option. The exported Hugging Face model returns one logit per encoded pair, so the caller is responsible for ranking the three candidate options.

## Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(repo_id, trust_remote_code=True)

false_sentence = "The sun rises in the west."
options = [
    "The sun rises in the east.",
    "The sun sets in the west.",
    "The sun shines at night.",
]

encoded = tokenizer(
    [false_sentence] * len(options),
    options,
    padding=True,
    truncation=True,
    return_tensors="pt",
)

scores = model(**encoded).logits.squeeze(-1)
best_idx = int(scores.argmax().item())
print(best_idx)
```

## Limitations

- Custom architecture requiring `trust_remote_code=True`
- Partial implementation from an in-progress project
- Not pretrained on a large general corpus
- Evaluated only on the course competition setup
- No broad safety, bias, or robustness evaluation

This upload should be treated as an experimental checkpoint for a specific academic task rather than a polished foundation model.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Export the custom AttentionTypes model to a Hugging Face-compatible repo folder.")
    parser.add_argument("--output-dir", default=str(ROOT / "hf_export" / "attentiontypes"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_STATE_DICT))
    parser.add_argument("--hparams", default=str(DEFAULT_HPARAMS))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--tokenizer-meta", default=str(DEFAULT_TOKENIZER_META))
    parser.add_argument("--repo-id", default="your-username/attentiontypes-commonsense")
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def load_config(hparams_path: Path, tokenizer_meta_path: Path, tokenizer_path: Path) -> AttentionTypesConfig:
    with hparams_path.open() as f:
        hparams = json.load(f)
    with tokenizer_meta_path.open() as f:
        tok_meta = json.load(f)

    vocab_size = Tokenizer.from_file(str(tokenizer_path)).get_vocab_size()
    config = AttentionTypesConfig(
        vocab_size=vocab_size,
        num_labels=1,
        pad_token_id=tok_meta["pad_id"],
        bos_token_id=tok_meta["cls_id"],
        eos_token_id=tok_meta["eos_id"],
        sep_token_id=tok_meta["sep_id"],
        unk_token_id=tok_meta["unk_id"],
        model_dim=hparams["model_dim"],
        num_heads=hparams["num_heads"],
        num_layers=hparams["num_layers"],
        ff_mult=hparams["ff_mult"],
        dropout=hparams["dropout"],
        max_position_embeddings=hparams["max_len"],
        pooling=hparams["pooling"],
        grouped_max_len=hparams["grouped_max_len"],
        bbpe_vocab_size=hparams["bbpe_vocab_size"],
        bbpe_min_freq=hparams["bbpe_min_freq"],
        ce_weight=hparams.get("ce_weight", 1.0),
        hinge_weight=hparams.get("hinge_weight", 0.0),
        hinge_margin=hparams.get("hinge_margin", 0.0),
        architectures=["AttentionTypesForSequenceClassification"],
        auto_map={
            "AutoConfig": "configuration_attentiontypes.AttentionTypesConfig",
            "AutoModelForSequenceClassification": "modeling_attentiontypes.AttentionTypesForSequenceClassification",
        },
    )
    return config


def build_tokenizer(tokenizer_path: Path, config: AttentionTypesConfig) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        eos_token="<eos>",
        bos_token="<cls>",
        model_max_length=config.grouped_max_len,
    )


def export_model(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.hparams), Path(args.tokenizer_meta), Path(args.tokenizer))
    model = AttentionTypesForSequenceClassification(config)

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. Missing={missing} Unexpected={unexpected}")

    tokenizer = build_tokenizer(Path(args.tokenizer), config)

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    shutil.copy2(HF_CODE_DIR / "configuration_attentiontypes.py", output_dir / "configuration_attentiontypes.py")
    shutil.copy2(HF_CODE_DIR / "modeling_attentiontypes.py", output_dir / "modeling_attentiontypes.py")

    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    for name in DATASET_FILES:
        shutil.copy2(DATA_DIR / name, dataset_dir / name)

    readme = MODEL_CARD_TEMPLATE.format(
        repo_name=args.repo_id.split("/")[-1],
        repo_id=args.repo_id,
        dataset_repo_id=args.dataset_repo_id,
        vocab_size=config.vocab_size,
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_mult=config.ff_mult,
        dropout=config.dropout,
        pooling=config.pooling,
        max_position_embeddings=config.max_position_embeddings,
        grouped_max_len=config.grouped_max_len,
        bbpe_min_freq=config.bbpe_min_freq,
    )
    (output_dir / "README.md").write_text(readme)

    loaded_config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    loaded_model = AutoModelForSequenceClassification.from_pretrained(output_dir, trust_remote_code=True)
    if loaded_config.model_type != "attentiontypes":
        raise RuntimeError(f"Unexpected model_type after export: {loaded_config.model_type}")
    if loaded_model.config.vocab_size != config.vocab_size:
        raise RuntimeError("Export verification failed: vocab size mismatch")

    if args.push:
        api = HfApi()
        api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(output_dir),
            commit_message="Upload custom AttentionTypes classifier",
        )

    return output_dir


def main():
    args = parse_args()
    output_dir = export_model(args)
    print(f"Exported Hugging Face package to: {output_dir}")
    if args.push:
        print(f"Pushed to Hugging Face Hub: {args.repo_id}")
    else:
        print("Push skipped. Re-run with --push once you are authenticated with Hugging Face.")


if __name__ == "__main__":
    main()
