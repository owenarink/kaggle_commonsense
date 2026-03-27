from __future__ import annotations

import argparse
import contextlib
import importlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src.analysis.architecture_viz_utils import OUTPUT_ROOT, ModelVizSpec, ensure_dir, selected_specs


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


def _build_pass_through_layer(keras, name: str):
    class PassThrough(keras.layers.Layer):
        def call(self, inputs):
            return inputs

    return PassThrough(name=name)


def build_surrogate_model(spec: ModelVizSpec):
    keras = _keras_modules()
    if keras is None:
        return None
    input_layer = keras.Input(shape=(32,), name="input_tokens")
    x = input_layer
    for index, layer_spec in enumerate(spec.layers[1:-1], start=1):
        x = _build_pass_through_layer(keras, f"{index:02d}_{layer_spec.name.lower().replace(' ', '_').replace('/', '_')}")(x)
    outputs = _build_pass_through_layer(keras, spec.layers[-1].name.lower().replace(" ", "_"))(x)
    return keras.Model(inputs=input_layer, outputs=outputs, name=spec.key)


def _draw_fallback(spec: ModelVizSpec, output_path: Path):
    fig_width = max(10, len(spec.layers) * 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_title(f"{spec.title}\nKeras-style fallback", fontsize=16, pad=16, color="#1D3557")

    positions = []
    y = 0.5
    for idx, layer in enumerate(spec.layers):
        x = 1.4 + idx * 2.0
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
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.35,rounding_size=0.08",
                "facecolor": "#F8F9FA",
                "edgecolor": "#6C757D",
                "linewidth": 1.2,
            },
        )

    for (x1, y1), (x2, y2) in zip(positions[:-1], positions[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.55, y1),
                (x2 - 0.55, y2),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color="#6C757D",
            )
        )

    ax.set_xlim(0.4, positions[-1][0] + 1.4)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_model(spec: ModelVizSpec, output_dir: Path):
    keras = _keras_modules()
    output_path = output_dir / f"{spec.key}_keras_basic.png"
    model = build_surrogate_model(spec)
    if keras is not None and model is not None:
        keras.utils.plot_model(
            model,
            to_file=str(output_path),
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            expand_nested=True,
            dpi=180,
        )
    else:
        _draw_fallback(spec, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Render simple keras.utils.plot_model diagrams for each model.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model keys to render.")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT / "architectures" / "keras_basic"))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    for spec in selected_specs(args.models):
        render_model(spec, output_dir)
        print(f"[ok] {spec.key} -> {output_dir / f'{spec.key}_keras_basic.png'}")


if __name__ == "__main__":
    main()
