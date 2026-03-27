from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import torch

from src.analysis.architecture_viz_utils import OUTPUT_ROOT, ensure_dir
from src.analysis.plot_loss_landscapes import (
    build_attentiontypes_transformer,
    build_cnn_pairwise,
    build_counterfact_transformer,
    build_cross_option_transformer,
    build_lec_transformer,
    build_mlp_multiclass,
    build_mlp_pairwise,
    build_plain_transformer,
    build_relsimple_transformer,
    build_sepdist_transformer,
)


def _netron_module():
    try:
        import netron
    except Exception:
        return None
    return netron


MODEL_EXPORTS = {
    "mlp_tfidf": (lambda: build_mlp_multiclass({"n_inputs": 20000}), torch.randn(1, 20000)),
    "mlp_pairwise": (lambda: build_mlp_pairwise({"n_inputs": 20000}), torch.randn(1, 20000)),
    "cnn_pairwise": (lambda: build_cnn_pairwise({"vocab_size": 50000}), torch.randint(0, 50000, (1, 128), dtype=torch.long)),
    "transformer_bbpe": (lambda: build_plain_transformer({"vocab_size": 12000}), torch.randint(0, 12000, (3, 128), dtype=torch.long)),
    "transformer_attentiontypes": (lambda: build_attentiontypes_transformer({"vocab_size": 12000}), torch.randint(0, 12000, (3, 128), dtype=torch.long)),
    "transformer_attention_relsimple": (lambda: build_relsimple_transformer({"vocab_size": 12000}), torch.randint(0, 12000, (3, 128), dtype=torch.long)),
    "transformer_attention_sepdist": (lambda: build_sepdist_transformer({"vocab_size": 12000}), torch.randint(0, 12000, (3, 128), dtype=torch.long)),
    "transformer_bertcounterfact": (lambda: build_counterfact_transformer({"vocab_size": 12000}), torch.randint(0, 12000, (3, 128), dtype=torch.long)),
    "transformer_bertcounter_cross_option": (lambda: build_cross_option_transformer({"vocab_size": 12000}), torch.randint(0, 12000, (1, 3, 128), dtype=torch.long)),
    "transformer_bertcounter_lec": (lambda: build_lec_transformer({"vocab_size": 12000}), torch.randint(0, 12000, (1, 3, 128), dtype=torch.long)),
}


def render_onnx_preview(model_name: str, output_dir: Path):
    if model_name in {"mlp_tfidf", "mlp_pairwise"}:
        labels = ["Input\n[1 x 20000]", "Dense\n[1 x 64]", "Output"]
    elif model_name == "cnn_pairwise":
        labels = ["Tokens\n[1 x 128]", "Embedding\n[1 x 128 x 128]", "ResNet CNN", "Pool", "Output"]
    elif model_name == "transformer_attentiontypes":
        labels = ["Input\n[3 x 128]", "Embedding", "Relative Pos", "Disentangled Attn x4", "Pool", "Head"]
    else:
        labels = ["Input", "Backbone", "Head", "Output"]

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 2.0), 3.8))
    fig.patch.set_facecolor("#F8F9FB")
    ax.set_facecolor("#F8F9FB")
    ax.set_title(f"{model_name} ONNX / Netron preview", fontsize=16, color="#1D3557", pad=16)

    xs = []
    x = 1.3
    for label in labels:
        xs.append(x)
        ax.text(
            x,
            0.9,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.35,rounding_size=0.08",
                "facecolor": "#E9F1F7",
                "edgecolor": "#355070",
                "linewidth": 1.4,
            },
        )
        x += 2.1

    for x1, x2 in zip(xs[:-1], xs[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.72, 0.9),
                (x2 - 0.72, 0.9),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.6,
                color="#6C757D",
            )
        )

    ax.axis("off")
    ax.set_xlim(0.4, x + 0.8)
    ax.set_ylim(0.0, 1.8)
    fig.tight_layout()
    png_path = output_dir / f"{model_name}_netron_preview.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return png_path


def parse_args():
    parser = argparse.ArgumentParser(description="Export ONNX models for Netron and optionally launch Netron.")
    parser.add_argument("--models", nargs="*", default=["transformer_attentiontypes"], help="Models to export.")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT / "architectures" / "netron"))
    parser.add_argument("--launch", action="store_true", help="Launch Netron server after export if available.")
    parser.add_argument("--port", type=int, default=8081, help="Netron port.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    exported = []

    for model_name in args.models:
        if model_name not in MODEL_EXPORTS:
            print(f"[skip] unknown model: {model_name}")
            continue
        model_ctor, dummy_input = MODEL_EXPORTS[model_name]
        model = model_ctor()
        model.eval()
        output_path = output_dir / f"{model_name}.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=14,
            dynamic_axes=None,
        )
        exported.append(output_path)
        preview_path = render_onnx_preview(model_name, output_dir)
        print(f"[ok] exported -> {output_path}")
        print(f"[ok] preview  -> {preview_path}")

    if args.launch and exported:
        netron = _netron_module()
        if netron is None:
            print("Netron python package not installed. Exported ONNX files only.")
            return
        netron.start([str(path) for path in exported], port=args.port, browse=True)


if __name__ == "__main__":
    main()
