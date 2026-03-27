from __future__ import annotations

import argparse
import contextlib
import importlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src.analysis.architecture_viz_utils import OUTPUT_ROOT, ensure_dir, selected_specs
from src.analysis.plot_architectures_keras_basic import build_surrogate_model


def _keras_modules():
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer), contextlib.redirect_stdout(buffer):
        try:
            return importlib.import_module("tensorflow").keras
        except Exception:
            try:
                return importlib.import_module("keras")
            except Exception:
                return None


def _draw_pretty_fallback(spec, output_path: Path):
    fig_width = max(11, len(spec.layers) * 2.15)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    fig.patch.set_facecolor("#F7F3EB")
    ax.set_facecolor("#F7F3EB")
    ax.set_title(spec.title, fontsize=18, color="#1D3557", pad=18, fontweight="bold")

    positions = []
    y = 0.62
    for idx, layer in enumerate(spec.layers):
        x = 1.6 + idx * 2.15
        positions.append((x, y))
        label = layer.name
        if layer.shape:
            label += f"\n{layer.shape}"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#243447",
            bbox={
                "boxstyle": "round,pad=0.5,rounding_size=0.12",
                "facecolor": "#FFF7E8",
                "edgecolor": "#355070",
                "linewidth": 1.8,
            },
        )

    for (x1, y1), (x2, y2) in zip(positions[:-1], positions[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.68, y1),
                (x2 - 0.68, y2),
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.8,
                color="#355070",
                connectionstyle="arc3,rad=0.0",
            )
        )

    if getattr(spec, "focus_points", ()):
        notes = "\n".join(f"- {point}" for point in spec.focus_points)
        ax.text(
            positions[len(positions) // 2][0],
            0.12,
            notes,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#5A3E2B",
            bbox={
                "boxstyle": "round,pad=0.45,rounding_size=0.12",
                "facecolor": "#FEFAE0",
                "edgecolor": "#BC6C25",
                "linewidth": 1.2,
            },
        )

    ax.set_xlim(0.5, positions[-1][0] + 1.6)
    ax.set_ylim(0.0, 1.05)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_styled_model(spec, output_dir: Path):
    keras = _keras_modules()
    model = build_surrogate_model(spec)
    output_path = output_dir / f"{spec.key}_keras_pretty.png"
    if keras is not None and model is not None:
        dot = keras.utils.model_to_dot(
            model,
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="LR",
            expand_nested=True,
            dpi=220,
        )
        dot.set_bgcolor("#F7F3EB")
        dot.set_rankdir("LR")
        dot.set_splines("ortho")
        dot.set_nodesep("0.45")
        dot.set_ranksep("0.7")
        dot.set_label(spec.title)
        dot.set_labelloc("t")
        dot.set_fontname("Helvetica-Bold")
        dot.set_fontsize("22")

        for node in dot.get_nodes():
            node.set_shape("record")
            node.set_style("rounded,filled")
            node.set_fillcolor("#FFF7E8")
            node.set_color("#355070")
            node.set_penwidth("1.7")
            node.set_fontname("Helvetica")
            node.set_fontsize("12")
            node.set_margin("0.22,0.12")

        if hasattr(dot, "write_png"):
            dot.write_png(str(output_path))
        else:
            dot.write(str(output_path.with_suffix(".dot")))
    else:
        _draw_pretty_fallback(spec, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Render higher-aesthetic keras model diagrams.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model keys to render.")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT / "architectures" / "keras_pretty"))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    for spec in selected_specs(args.models):
        render_styled_model(spec, output_dir)
        print(f"[ok] {spec.key} -> {output_dir / f'{spec.key}_keras_pretty.png'}")


if __name__ == "__main__":
    main()
