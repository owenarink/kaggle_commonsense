import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

if os.environ.get("MPLBACKEND", "") == "":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.models.bert import BertTransformer
from src.models.bert_attention_experiments import BertRelSimpleTransformer, BertSepDistanceTransformer
from src.models.bertcounter_latent_edit_competition import BertCounterFactLatentEditCompetitionTransformer
from src.models.bertcounterfact import BertCounterFactTransformer
from src.models.bertcounterfact_cross_option_competition import BertCounterFactCrossOpitionCompetitionTransformer
from src.models.cnn_text import TextResNetCNN
from src.models.mlp_tfidf import MLP
from src.models.transformer import TextTransformer
from src.processing import preprocess
from src.train_model import load_prep_data as load_mlp_multiclass_data
from src.train_model_cnn import load_prep_data_pairwise as load_cnn_pairwise_data
from src.train_model_transformer import preprocess_transformer_grouped_bbpe as load_plain_transformer_data
from src.train_model_transformer_attention_relsimple import (
    preprocess_transformer_grouped_bbpe as load_relsimple_transformer_data,
)
from src.train_model_transformer_attention_sepdist import (
    preprocess_transformer_grouped_bbpe as load_sepdist_transformer_data,
)
from src.train_model_transformer_attentiontypes import (
    preprocess_transformer_grouped_bbpe as load_attentiontypes_transformer_data,
)
from src.train_model_wPairwise import load_prep_data_pairwise as load_mlp_pairwise_data
from src.train_model_bertcounterfact import preprocess_transformer_grouped_bbpe as load_counterfact_data
from src.train_model_bertcounterfact_cross_option import (
    preprocess_transformer_grouped_bbpe as load_cross_option_data,
)
from src.train_model_bertcounterfact_latent_edit_competition import (
    preprocess_transformer_grouped_bbpe as load_lec_data,
)


torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "plots" / "loss_landscapes"
CHECKPOINT_DIR = ROOT / "checkpoints"
DEFAULT_SEED = 40


@dataclass
class ModelSpec:
    name: str
    checkpoint_candidates: list[str]
    loader: Callable[[], tuple[torch.Tensor, torch.Tensor, dict]]
    builder: Callable[[dict], nn.Module]
    plot_title: str


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def first_existing(paths: list[str]) -> str | None:
    for path in paths:
        if Path(path).exists():
            return path
    return None


def load_json(path: str) -> dict:
    with open(path, "r") as handle:
        return json.load(handle)


def prepare_train_df():
    import pandas as pd

    x_train_raw = pd.read_csv(ROOT / "data" / "train_data.csv")
    x_test_raw = pd.read_csv(ROOT / "data" / "test_data.csv")
    y_train_raw = pd.read_csv(ROOT / "data" / "train_answers.csv")
    train_df, _ = preprocess(x_train_raw, x_test_raw, y_train_raw)
    return train_df


def load_mlp_multiclass_tensors(max_samples: int | None = None):
    _, x_val, _, y_val = load_mlp_multiclass_data()
    x = torch.tensor(x_val, dtype=torch.float32)
    y = torch.tensor(y_val, dtype=torch.long)
    if max_samples is not None:
        x = x[:max_samples]
        y = y[:max_samples]
    meta = {"n_inputs": int(x.shape[1]), "loss_name": "CrossEntropyLoss"}
    return x, y, meta


def load_mlp_pairwise_tensors(max_samples: int | None = None):
    _, x_val, _, y_val, _, _, _ = load_mlp_pairwise_data(seed=DEFAULT_SEED)
    x = torch.tensor(x_val, dtype=torch.float32)
    y = torch.tensor(y_val, dtype=torch.float32)
    if max_samples is not None:
        x = x[:max_samples]
        y = y[:max_samples]
    meta = {"n_inputs": int(x.shape[1]), "loss_name": "BCEWithLogitsLoss"}
    return x, y, meta


def load_cnn_pairwise_tensors(max_samples: int | None = None):
    _, x_val, _, y_val, _, _, _, vocab = load_cnn_pairwise_data(seed=DEFAULT_SEED)
    x = torch.tensor(x_val, dtype=torch.long)
    y = torch.tensor(y_val, dtype=torch.float32)
    if max_samples is not None:
        x = x[:max_samples]
        y = y[:max_samples]
    meta = {"vocab_size": int(len(vocab)), "loss_name": "BCEWithLogitsLoss"}
    return x, y, meta


def load_grouped_tensors(
    loader_fn: Callable,
    max_samples: int | None = None,
):
    train_df = prepare_train_df()
    _, x_val, _, y_val, tok_info = loader_fn(train_df=train_df, seed=DEFAULT_SEED, max_len=128)
    x = torch.tensor(x_val, dtype=torch.long)
    y = torch.tensor(y_val, dtype=torch.long)
    if max_samples is not None:
        x = x[:max_samples]
        y = y[:max_samples]
    meta = {
        "vocab_size": int(tok_info["vocab_size"]),
        "loss_name": "CrossEntropyLoss(label_smoothing=0.10)",
    }
    return x, y, meta


def mlp_multiclass_spec(max_samples: int | None):
    return load_mlp_multiclass_tensors(max_samples=max_samples)


def mlp_pairwise_spec(max_samples: int | None):
    return load_mlp_pairwise_tensors(max_samples=max_samples)


def cnn_pairwise_spec(max_samples: int | None):
    return load_cnn_pairwise_tensors(max_samples=max_samples)


def grouped_spec(loader_fn: Callable, max_samples: int | None):
    return load_grouped_tensors(loader_fn=loader_fn, max_samples=max_samples)


def build_mlp_multiclass(meta: dict):
    return MLP(n_inputs=meta["n_inputs"], n_hidden=[64], n_classes=3)


def build_mlp_pairwise(meta: dict):
    return MLP(n_inputs=meta["n_inputs"], n_hidden=[64], n_classes=1)


def build_cnn_pairwise(meta: dict):
    return TextResNetCNN(vocab_size=meta["vocab_size"], num_classes=1, pad_idx=0)


def build_plain_transformer(meta: dict):
    hp = load_json(CHECKPOINT_DIR / "transformer_bbpe_hparams.json")
    return TextTransformer(
        vocab_size=meta["vocab_size"],
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


def build_attentiontypes_transformer(meta: dict):
    hp = load_json(CHECKPOINT_DIR / "transformer_attentiontypes_hparams.json")
    return BertTransformer(
        vocab_size=meta["vocab_size"],
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


def build_relsimple_transformer(meta: dict):
    hp = load_json(CHECKPOINT_DIR / "transformer_attention_relsimple_hparams.json")
    return BertRelSimpleTransformer(
        vocab_size=meta["vocab_size"],
        num_classes=1,
        pad_idx=int(hp["pad_idx"]),
        sep_idx=int(hp["sep_idx"]),
        model_dim=int(hp["model_dim"]),
        num_heads=int(hp["num_heads"]),
        num_layers=int(hp["num_layers"]),
        ff_mult=int(hp["ff_mult"]),
        dropout=float(hp["dropout"]),
        max_len=int(hp["max_len"]),
        pooling=str(hp["pooling"]),
    )


def build_sepdist_transformer(meta: dict):
    hp = load_json(CHECKPOINT_DIR / "transformer_attention_sepdist_hparams.json")
    return BertSepDistanceTransformer(
        vocab_size=meta["vocab_size"],
        num_classes=1,
        pad_idx=int(hp["pad_idx"]),
        sep_idx=int(hp["sep_idx"]),
        model_dim=int(hp["model_dim"]),
        num_heads=int(hp["num_heads"]),
        num_layers=int(hp["num_layers"]),
        ff_mult=int(hp["ff_mult"]),
        dropout=float(hp["dropout"]),
        max_len=int(hp["max_len"]),
        pooling=str(hp["pooling"]),
    )


def build_counterfact_transformer(meta: dict):
    hp = load_json(CHECKPOINT_DIR / "transformer_bertcounterfact_hparams.json")
    return BertCounterFactTransformer(
        vocab_size=meta["vocab_size"],
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


def build_cross_option_transformer(meta: dict):
    hp = load_json(CHECKPOINT_DIR / "transformer_bertcounter_cross_option_hparams.json")
    return BertCounterFactCrossOpitionCompetitionTransformer(
        vocab_size=meta["vocab_size"],
        pad_idx=int(hp["pad_idx"]),
        model_dim=int(hp["model_dim"]),
        num_heads=int(hp["num_heads"]),
        num_layers=int(hp["num_layers"]),
        ff_mult=int(hp["ff_mult"]),
        dropout=float(hp["dropout"]),
        max_len=int(hp["max_len"]),
        pooling=str(hp["pooling"]),
    )


def build_lec_transformer(meta: dict):
    hp = load_json(CHECKPOINT_DIR / "transformer_bertcounter_LEC_hparams.json")
    return BertCounterFactLatentEditCompetitionTransformer(
        vocab_size=meta["vocab_size"],
        pad_idx=int(hp["pad_idx"]),
        sep_idx=int(hp["sep_idx"]),
        model_dim=int(hp["model_dim"]),
        num_heads=int(hp["num_heads"]),
        num_layers=int(hp["num_layers"]),
        ff_mult=int(hp["ff_mult"]),
        dropout=float(hp["dropout"]),
        max_len=int(hp["max_len"]),
        pooling=str(hp["pooling"]),
        edit_minimality_weight=float(hp["edit_minimality_weight"]),
    )


def _is_bias_or_norm(param: torch.Tensor) -> bool:
    return param.ndim <= 1


def _normalize_like_goldstein(delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    if _is_bias_or_norm(base):
        return torch.zeros_like(base)

    out = delta.clone()
    out_flat = out.reshape(out.shape[0], -1)
    base_flat = base.reshape(base.shape[0], -1)
    delta_norms = torch.norm(out_flat, dim=1, keepdim=True)
    base_norms = torch.norm(base_flat, dim=1, keepdim=True)
    scales = torch.where(delta_norms > 0, base_norms / delta_norms.clamp(min=1e-12), torch.zeros_like(delta_norms))
    scaled = out_flat * scales
    return scaled.reshape_as(base)


def create_random_direction(base_params: list[torch.Tensor]):
    return [_normalize_like_goldstein(torch.randn_like(base), base) for base in base_params]


def orthogonalize(direction: list[torch.Tensor], reference: list[torch.Tensor]):
    numerator = sum(torch.sum(d * r) for d, r in zip(direction, reference))
    denominator = sum(torch.sum(r * r) for r in reference)
    if float(denominator) == 0.0:
        return direction
    scale = numerator / denominator
    return [d - scale * r for d, r in zip(direction, reference)]


def set_parameters(model: nn.Module, base_params: list[torch.Tensor], dir_x: list[torch.Tensor], dir_y: list[torch.Tensor], alpha: float, beta: float):
    with torch.no_grad():
        for param, base, dx, dy in zip(model.parameters(), base_params, dir_x, dir_y):
            if not param.requires_grad:
                continue
            param.copy_(base + alpha * dx + beta * dy)


def evaluate_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        if y.dtype == torch.long:
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            if logits.ndim == 2 and logits.shape[0] == x.shape[0]:
                if logits.shape[1] == 1:
                    raise ValueError("Cross-entropy targets require at least 2 logits per sample.")
                loss = nn.CrossEntropyLoss(label_smoothing=0.10)(logits, y)
            else:
                batch_size, num_options, seq_len = x.shape
                del seq_len
                scores = logits.view(batch_size, num_options)
                loss = nn.CrossEntropyLoss(label_smoothing=0.10)(scores, y)
        else:
            logits = logits.view(-1)
            loss = nn.BCEWithLogitsLoss()(logits, y.view(-1))
    return float(loss.item())


def evaluate_grouped_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, grouped_direct: bool) -> float:
    model.eval()
    with torch.no_grad():
        if grouped_direct:
            scores = model(x)
        else:
            batch_size, num_options, seq_len = x.shape
            del seq_len
            scores = model(x.view(batch_size * num_options, -1)).view(batch_size, num_options)
        loss = nn.CrossEntropyLoss(label_smoothing=0.10)(scores, y)
    return float(loss.item())


def plot_landscape(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    output_path: Path,
    title: str,
    span: float,
    grid_size: int,
    grouped_direct: bool,
    checkpoint_label: str,
    loss_name: str,
):
    params = [param for param in model.parameters() if param.requires_grad]
    base_params = [param.detach().clone() for param in params]

    dir_x = create_random_direction(base_params)
    dir_y = create_random_direction(base_params)
    dir_y = orthogonalize(dir_y, dir_x)
    dir_y = [_normalize_like_goldstein(delta, base) for delta, base in zip(dir_y, base_params)]

    alphas = np.linspace(-span, span, grid_size)
    betas = np.linspace(-span, span, grid_size)
    landscape = np.zeros((grid_size, grid_size), dtype=np.float32)

    for row, beta in enumerate(betas):
        for col, alpha in enumerate(alphas):
            set_parameters(model, base_params, dir_x, dir_y, float(alpha), float(beta))
            if x.ndim == 3:
                landscape[row, col] = evaluate_grouped_loss(model, x, y, grouped_direct=grouped_direct)
            else:
                landscape[row, col] = evaluate_loss(model, x, y)

    set_parameters(model, base_params, dir_x, dir_y, 0.0, 0.0)
    x_grid, y_grid = np.meshgrid(alphas, betas)
    checkpoint_loss = landscape[grid_size // 2, grid_size // 2]

    np.savez_compressed(
        output_path.with_suffix(".npz"),
        x=x_grid,
        y=y_grid,
        z=landscape,
        checkpoint=checkpoint_label,
        loss_name=loss_name,
        title=title,
    )

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        x_grid,
        y_grid,
        landscape,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
        rstride=1,
        cstride=1,
    )
    ax.contour(
        x_grid,
        y_grid,
        landscape,
        zdir="z",
        offset=float(np.min(landscape)),
        levels=12,
        cmap="viridis",
        linewidths=0.8,
    )
    ax.scatter([0.0], [0.0], [checkpoint_loss], color="crimson", s=55, depthshade=False)
    ax.text(0.0, 0.0, checkpoint_loss, " checkpoint", color="crimson")
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_zlabel(loss_name)
    ax.set_title(title)
    ax.view_init(elev=35, azim=-60)
    fig.colorbar(surface, ax=ax, shrink=0.68, pad=0.08, label=loss_name)
    ax.text2D(
        0.02,
        0.03,
        checkpoint_label,
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.35, "pad": 4},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_state_dict(model: nn.Module, checkpoint_path: str, device: torch.device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict)
    return model


def build_registry(max_samples: int | None):
    return [
        ModelSpec(
            name="mlp_tfidf",
            checkpoint_candidates=[
                str(CHECKPOINT_DIR / "mlp_tfidf.pt"),
            ],
            loader=lambda: mlp_multiclass_spec(max_samples),
            builder=build_mlp_multiclass,
            plot_title="Loss Landscape: MLP TF-IDF",
        ),
        ModelSpec(
            name="mlp_pairwise",
            checkpoint_candidates=[
                str(CHECKPOINT_DIR / "mlp_pairwise.pt"),
            ],
            loader=lambda: mlp_pairwise_spec(max_samples),
            builder=build_mlp_pairwise,
            plot_title="Loss Landscape: MLP Pairwise",
        ),
        ModelSpec(
            name="cnn_pairwise",
            checkpoint_candidates=[
                str(CHECKPOINT_DIR / "cnn_pairwise.pt"),
            ],
            loader=lambda: cnn_pairwise_spec(max_samples),
            builder=build_cnn_pairwise,
            plot_title="Loss Landscape: CNN Pairwise",
        ),
        ModelSpec(
            name="transformer_bbpe",
            checkpoint_candidates=[
                str(CHECKPOINT_DIR / "transformer_best_loss.pt"),
                str(CHECKPOINT_DIR / "transformer_best_acc.pt"),
                str(CHECKPOINT_DIR / "transformer_pairwise.pt"),
            ],
            loader=lambda: grouped_spec(load_plain_transformer_data, max_samples),
            builder=build_plain_transformer,
            plot_title="Loss Landscape: Transformer BBPE",
        ),
        ModelSpec(
            name="transformer_attentiontypes",
            checkpoint_candidates=[
                str(CHECKPOINT_DIR / "transformer_attentiontypes_best_loss.pt"),
                str(CHECKPOINT_DIR / "transformer_attentiontypes_best_acc.pt"),
                str(CHECKPOINT_DIR / "transformer_attentiontypes.pt"),
            ],
            loader=lambda: grouped_spec(load_attentiontypes_transformer_data, max_samples),
            builder=build_attentiontypes_transformer,
            plot_title="Loss Landscape: Attention Types",
        ),
        ModelSpec(
            name="transformer_attention_relsimple",
            checkpoint_candidates=[
                str(CHECKPOINT_DIR / "transformer_attention_relsimple_best_loss.pt"),
                str(CHECKPOINT_DIR / "transformer_attention_relsimple_best_acc.pt"),
                str(CHECKPOINT_DIR / "transformer_attention_relsimple.pt"),
            ],
            loader=lambda: grouped_spec(load_relsimple_transformer_data, max_samples),
            builder=build_relsimple_transformer,
            plot_title="Loss Landscape: RelSimple Attention",
        ),
        ModelSpec(
            name="transformer_attention_sepdist",
            checkpoint_candidates=[
                str(CHECKPOINT_DIR / "transformer_attention_sepdist_best_loss.pt"),
                str(CHECKPOINT_DIR / "transformer_attention_sepdist_best_acc.pt"),
                str(CHECKPOINT_DIR / "transformer_attention_sepdist.pt"),
            ],
            loader=lambda: grouped_spec(load_sepdist_transformer_data, max_samples),
            builder=build_sepdist_transformer,
            plot_title="Loss Landscape: SEP Distance Attention",
        ),
        ModelSpec(
            name="transformer_bertcounterfact",
            checkpoint_candidates=[
                str(CHECKPOINT_DIR / "transformer_bertcounterfact_best_loss.pt"),
                str(CHECKPOINT_DIR / "transformer_bertcounterfact_best_acc.pt"),
                str(CHECKPOINT_DIR / "transformer_bertcounterfact.pt"),
            ],
            loader=lambda: grouped_spec(load_counterfact_data, max_samples),
            builder=build_counterfact_transformer,
            plot_title="Loss Landscape: CounterFact Transformer",
        ),
        ModelSpec(
            name="transformer_bertcounter_cross_option",
            checkpoint_candidates=[
                str(CHECKPOINT_DIR / "transformer_bertcounter_cross_option_best_loss.pt"),
                str(CHECKPOINT_DIR / "transformer_bertcounter_cross_option_best_acc.pt"),
                str(CHECKPOINT_DIR / "transformer_bertcounter_cross_option.pt"),
            ],
            loader=lambda: grouped_spec(load_cross_option_data, max_samples),
            builder=build_cross_option_transformer,
            plot_title="Loss Landscape: Cross Option CounterFact",
        ),
        ModelSpec(
            name="transformer_bertcounter_lec",
            checkpoint_candidates=[
                str(CHECKPOINT_DIR / "transformer_bertcounter_LEC_best_loss.pt"),
                str(CHECKPOINT_DIR / "transformer_bertcounter_LEC_best_acc.pt"),
                str(CHECKPOINT_DIR / "transformer_bertcounter_LEC.pt"),
            ],
            loader=lambda: grouped_spec(load_lec_data, max_samples),
            builder=build_lec_transformer,
            plot_title="Loss Landscape: Latent Edit Competition",
        ),
    ]


def grouped_direct_model(name: str) -> bool:
    return name in {
        "transformer_bertcounter_cross_option",
        "transformer_bertcounter_lec",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Plot checkpoint loss landscapes into outputs/plots/loss_landscapes.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model names to plot.")
    parser.add_argument("--grid-size", type=int, default=21, help="Number of points per axis in the loss grid.")
    parser.add_argument("--span", type=float, default=1.0, help="Range of each random parameter direction.")
    parser.add_argument("--max-samples", type=int, default=96, help="Cap validation samples per model for faster plotting.")
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto", help="Execution device.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    supported_specs = build_registry(max_samples=args.max_samples)
    selected = set(args.models) if args.models else None

    produced = []
    skipped = []

    for spec in supported_specs:
        if selected and spec.name not in selected:
            continue

        checkpoint_path = first_existing(spec.checkpoint_candidates)
        if checkpoint_path is None:
            skipped.append((spec.name, "missing checkpoint"))
            continue

        try:
            x, y, meta = spec.loader()
            model = spec.builder(meta).to(device)
            model = load_state_dict(model, checkpoint_path, device=device)
            x = x.to(device)
            y = y.to(device)

            output_path = OUTPUT_DIR / f"{spec.name}_loss_landscape.png"
            plot_landscape(
                model=model,
                x=x,
                y=y,
                output_path=output_path,
                title=spec.plot_title,
                span=args.span,
                grid_size=args.grid_size,
                grouped_direct=grouped_direct_model(spec.name),
                checkpoint_label=Path(checkpoint_path).name,
                loss_name=meta["loss_name"],
            )
            produced.append(output_path)
            print(f"[ok] {spec.name} -> {output_path}")
        except Exception as exc:
            skipped.append((spec.name, str(exc)))
            print(f"[skip] {spec.name}: {exc}")

    unsupported = []
    segment_bias_ckpt = CHECKPOINT_DIR / "transformer_attention_segment_bias.pt"
    if (selected is None or "transformer_attention_segment_bias" in selected) and segment_bias_ckpt.exists():
        unsupported.append(("transformer_attention_segment_bias", "checkpoint exists but no matching source model is present in this repo"))

    for name, reason in unsupported:
        print(f"[skip] {name}: {reason}")

    print(f"Generated {len(produced)} loss landscape plot(s) in {OUTPUT_DIR}")
    if skipped or unsupported:
        print("Skipped models:")
        for name, reason in [*skipped, *unsupported]:
            print(f" - {name}: {reason}")


if __name__ == "__main__":
    main()
