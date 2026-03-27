import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.models.bert import BertTransformer


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"
OUT_DIR = ROOT / "outputs" / "plots" / "readme"


def load_hparams():
    with open(CHECKPOINTS / "transformer_attentiontypes_hparams.json", "r") as f:
        return json.load(f)


def build_model(hp):
    return BertTransformer(
        vocab_size=int(hp["bbpe_vocab_size"]),
        num_classes=1,
        pad_idx=int(hp["pad_idx"]),
        model_dim=int(hp["model_dim"]),
        num_heads=int(hp["num_heads"]),
        num_layers=int(hp["num_layers"]),
        ff_mult=int(hp["ff_mult"]),
        dropout=float(hp["dropout"]),
        max_len=int(hp["max_len"]),
        pooling=str(hp["pooling"]),
    )


def group_name(param_name: str) -> str:
    if param_name.startswith("embedding."):
        return "Token embedding"
    if ".self_attn.rel_pos_emb." in param_name:
        return "Relative position embeddings"
    if ".self_attn." in param_name:
        return "Disentangled attention projections"
    if ".ff." in param_name:
        return "Feed-forward blocks"
    if ".norm" in param_name:
        return "LayerNorm"
    if param_name.startswith("out."):
        return "Output head"
    return "Other"


def plot_parameter_breakdown(model):
    counts = {}
    for name, param in model.named_parameters():
        group = group_name(name)
        counts[group] = counts.get(group, 0) + param.numel()

    labels = [
        "Token embedding",
        "Relative position embeddings",
        "Disentangled attention projections",
        "Feed-forward blocks",
        "LayerNorm",
        "Output head",
        "Other",
    ]
    values = [counts.get(label, 0) for label in labels]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = ["#5B6CFF", "#8C5EFA", "#D062A4", "#F28E2B", "#59A14F", "#4E79A7", "#9D9DA1"]
    bars = ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_title("AttentionTypes Parameter Breakdown", fontsize=15, weight="bold")
    ax.set_xlabel("Number of parameters")
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    for bar, value in zip(bars, values):
        pct = (100.0 * value / total) if total else 0.0
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{value/1e6:.2f}M ({pct:.1f}%)",
            va="center",
            fontsize=10,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attentiontypes_parameter_breakdown.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def lr_multiplier(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05) -> float:
    step = min(step, total_steps)
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def plot_lr_schedule():
    base_lr = 6e-5
    train_size = 6400
    batch_size = 32
    steps_per_epoch = train_size // batch_size
    schedule_epochs = 40
    total_steps = schedule_epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)

    steps = np.arange(total_steps + 1)
    lrs = np.array([base_lr * lr_multiplier(s, warmup_steps, total_steps) for s in steps])

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(steps, lrs, color="#6F42C1", linewidth=2.5)
    ax.axvline(warmup_steps, color="#D62728", linestyle="--", linewidth=1.5, label="Warmup ends")
    ax.set_title("AdamW Warmup-Cosine Learning Rate Schedule", fontsize=15, weight="bold")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Learning rate")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attentiontypes_lr_schedule.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hp = load_hparams()
    model = build_model(hp)
    plot_parameter_breakdown(model)
    plot_lr_schedule()


if __name__ == "__main__":
    main()
