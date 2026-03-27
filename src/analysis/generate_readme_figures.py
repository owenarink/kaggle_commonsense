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


def plot_loss_landscape_variants():
    data = np.load(ROOT / "outputs" / "plots" / "loss_landscapes" / "transformer_attentiontypes_loss_landscape.npz")
    x = data["x"]
    y = data["y"]
    z = data["z"].astype(float)

    zmin = np.nanmin(z)
    min_idx = np.unravel_index(np.nanargmin(z), z.shape)
    x0, y0, z0 = x[min_idx], y[min_idx], z[min_idx]

    fig = plt.figure(figsize=(8.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, z, cmap="magma", linewidth=0, antialiased=True, alpha=0.96)
    ax.contour(x, y, z, zdir="z", offset=zmin - 0.05, cmap="magma", levels=12, linewidths=1.1)
    ax.scatter([x0], [y0], [z0], color="#7CFF6B", s=55, depthshade=False)
    ax.set_title("AttentionTypes Loss Basin", fontsize=15, weight="bold")
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_zlabel("Validation loss")
    ax.view_init(elev=32, azim=-58)
    fig.colorbar(surf, shrink=0.62, pad=0.08)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attentiontypes_loss_basin.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    gy, gx = np.gradient(z)
    sharpness = np.sqrt(gx ** 2 + gy ** 2)
    fig = plt.figure(figsize=(8.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, sharpness, cmap="viridis", linewidth=0, antialiased=True, alpha=0.96)
    ax.contour(x, y, sharpness, zdir="z", offset=np.nanmin(sharpness) - 0.01, cmap="viridis", levels=12, linewidths=1.0)
    ax.set_title("AttentionTypes Sharpness Surface", fontsize=15, weight="bold")
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_zlabel("Gradient magnitude")
    ax.view_init(elev=28, azim=38)
    fig.colorbar(surf, shrink=0.62, pad=0.08)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attentiontypes_sharpness_surface.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_relative_position_surface():
    ckpt = np.load
    hp = load_hparams()
    model = build_model(hp)
    state = __import__("torch").load(CHECKPOINTS / "transformer_attentiontypes_best_acc.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)

    layer_norms = []
    for layer_idx, layer in enumerate(model.transformer.layers):
        emb = layer.self_attn.rel_pos_emb.weight.detach().cpu().numpy()
        norms = np.linalg.norm(emb, axis=1)
        layer_norms.append(norms)

    z = np.stack(layer_norms, axis=0)
    y = np.arange(z.shape[0])
    x = np.arange(z.shape[1]) - (z.shape[1] // 2)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8.4, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, z, cmap="plasma", linewidth=0, antialiased=True, alpha=0.97)
    ax.set_title("Relative Position Embedding Energy", fontsize=15, weight="bold")
    ax.set_xlabel("Relative offset")
    ax.set_ylabel("Encoder layer")
    ax.set_zlabel("Embedding norm")
    ax.view_init(elev=30, azim=-50)
    fig.colorbar(surf, shrink=0.62, pad=0.08)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attentiontypes_relative_position_surface.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_projection_energy_surface():
    import torch

    hp = load_hparams()
    model = build_model(hp)
    state = torch.load(CHECKPOINTS / "transformer_attentiontypes_best_acc.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)

    component_labels = ["q", "k", "v", "pos_k", "pos_q", "out", "ff1", "ff2"]
    z_rows = []
    for layer in model.transformer.layers:
        z_rows.append([
            float(layer.self_attn.q_proj.weight.norm().item()),
            float(layer.self_attn.k_proj.weight.norm().item()),
            float(layer.self_attn.v_proj.weight.norm().item()),
            float(layer.self_attn.pos_key_proj.weight.norm().item()),
            float(layer.self_attn.pos_query_proj.weight.norm().item()),
            float(layer.self_attn.o_proj.weight.norm().item()),
            float(layer.ff[0].weight.norm().item()),
            float(layer.ff[3].weight.norm().item()),
        ])

    z = np.array(z_rows, dtype=float)
    y = np.arange(z.shape[0])
    x = np.arange(z.shape[1])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8.8, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, z, cmap="cividis", linewidth=0, antialiased=True, alpha=0.97)
    ax.set_title("Projection Weight Energy Across Layers", fontsize=15, weight="bold")
    ax.set_xlabel("Projection block")
    ax.set_ylabel("Encoder layer")
    ax.set_zlabel("Weight norm")
    ax.set_xticks(x)
    ax.set_xticklabels(component_labels)
    ax.view_init(elev=28, azim=35)
    fig.colorbar(surf, shrink=0.62, pad=0.08)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attentiontypes_projection_energy_surface.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_offset_asymmetry_surface():
    import torch

    hp = load_hparams()
    model = build_model(hp)
    state = torch.load(CHECKPOINTS / "transformer_attentiontypes_best_acc.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)

    max_len = int(hp["max_len"])
    center = max_len - 1
    max_offset = min(64, max_len - 1)

    rows = []
    for layer in model.transformer.layers:
        emb = layer.self_attn.rel_pos_emb.weight.detach().cpu().numpy()
        vals = []
        for d in range(1, max_offset + 1):
            pos = emb[center + d]
            neg = emb[center - d]
            vals.append(float(np.linalg.norm(pos - neg)))
        rows.append(vals)

    z = np.array(rows, dtype=float)
    y = np.arange(z.shape[0])
    x = np.arange(1, z.shape[1] + 1)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8.6, 6.3))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, z, cmap="turbo", linewidth=0, antialiased=True, alpha=0.97)
    ax.set_title("Directional Offset Asymmetry", fontsize=15, weight="bold")
    ax.set_xlabel("|relative offset|")
    ax.set_ylabel("Encoder layer")
    ax.set_zlabel(r"$\|r(+d)-r(-d)\|_2$")
    ax.view_init(elev=30, azim=42)
    fig.colorbar(surf, shrink=0.62, pad=0.08)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attentiontypes_offset_asymmetry_surface.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hp = load_hparams()
    model = build_model(hp)
    plot_parameter_breakdown(model)
    plot_lr_schedule()
    plot_loss_landscape_variants()
    plot_relative_position_surface()
    plot_projection_energy_surface()
    plot_offset_asymmetry_surface()


if __name__ == "__main__":
    main()
