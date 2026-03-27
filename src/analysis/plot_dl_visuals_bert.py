from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from src.analysis.architecture_viz_utils import PALETTE


OUTPUT_DIR = Path("outputs/plots/dl_visuals")


COLORS = {
    "input": PALETTE["input"],
    "embed": PALETTE["embedding"],
    "pos": PALETTE["position"],
    "attn": PALETTE["attention"],
    "mix": PALETTE["competition"],
    "norm": PALETTE["norm"],
    "ffn": PALETTE["ffn"],
    "pool": PALETTE["pool"],
    "head": PALETTE["output"],
    "decision": PALETTE["decision"],
    "residual": "#6C757D",
}


def _box(ax, x, y, w, h, text, color, fontsize=11, radius=0.12, text_color="#132A3E", lw=1.6):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw,
        edgecolor="#2B2D42",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=text_color)


def _arrow(ax, start, end, color="#5C677D", lw=1.7, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def _stack_panel(ax, x, y, w, h, face="#EFEFEF"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor="none", zorder=0))


def _skip(ax, x0, y0, x1, y1, x2, y2, color="#6C757D"):
    ax.plot([x0, x1, x1, x2], [y0, y0, y1, y2], color=color, linewidth=1.25)


def _circle_node(ax, x, y, r, text, face="#DCD6F7", text_color="#5C677D"):
    circ = Circle((x, y), r, facecolor=face, edgecolor="#8D99AE", linewidth=1.0)
    ax.add_patch(circ)
    ax.text(x, y, text, ha="center", va="center", fontsize=11, color=text_color)


def render_bert_visual(output_path: Path):
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _stack_panel(ax, 5.2, 2.7, 3.8, 4.05)
    _stack_panel(ax, 9.9, 2.7, 4.2, 4.95)

    _box(ax, 10.42, 7.88, 2.62, 0.42, "Grouped Cross-Entropy\n[3 classes]", COLORS["decision"], fontsize=11.2, radius=0.08, text_color="white")
    _box(ax, 10.42, 7.28, 2.62, 0.48, "Linear Classification Head\n[3 x 1]", COLORS["head"], fontsize=10.8, radius=0.08, text_color="white")
    _box(ax, 10.48, 6.56, 2.5, 0.56, "Mean / CLS Pooling\n[3 x 256]", COLORS["pool"], fontsize=10.8, radius=0.08)

    _box(ax, 10.48, 5.96, 2.5, 0.44, "Add & Norm\n[3 x 128 x 256]", COLORS["norm"], fontsize=10.9, radius=0.08)
    _box(ax, 10.18, 5.0, 3.1, 0.82, "Position-wise Feed Forward\n[3 x 128 x 256]", COLORS["ffn"], fontsize=12.6, radius=0.10)
    _box(ax, 10.48, 4.38, 2.5, 0.44, "Add & Norm\n[3 x 128 x 256]", COLORS["norm"], fontsize=10.9, radius=0.08)
    _box(ax, 10.18, 3.16, 3.1, 1.02, "Disentangled Multi-Head\nSelf-Attention\n[3 x 128 x 256]", COLORS["mix"], fontsize=12.0, radius=0.10, text_color="white")
    _box(ax, 10.7, 2.52, 2.08, 0.42, "8 heads, x4 layers", COLORS["pos"], fontsize=10.4, radius=0.08)

    _box(ax, 5.92, 5.42, 2.48, 0.50, "Content-to-Content\n[3 x 128 x 256]", COLORS["attn"], fontsize=11.2, radius=0.08, text_color="white")
    _box(ax, 5.92, 4.54, 2.48, 0.50, "Content-to-Position\n[3 x 128 x 256]", COLORS["attn"], fontsize=11.2, radius=0.08, text_color="white")
    _box(ax, 5.92, 3.66, 2.48, 0.50, "Position-to-Content\n[3 x 128 x 256]", COLORS["attn"], fontsize=11.2, radius=0.08, text_color="white")

    _box(ax, 10.42, 0.98, 2.62, 0.76, "Input Embedding\n[3 x 128 x 256]", COLORS["embed"], fontsize=12.4, radius=0.10, text_color="white")

    _circle_node(ax, 9.55, 2.08, 0.23, "+")
    _circle_node(ax, 13.15, 2.08, 0.23, "+")
    _circle_node(ax, 8.65, 2.08, 0.38, "↶")
    _circle_node(ax, 14.05, 2.08, 0.38, "↷")

    ax.text(7.55, 2.08, "Relative Position\nEmbeddings", ha="center", va="center", fontsize=12, color="#8A8A8A")
    ax.text(15.05, 2.08, "Grouped Token IDs\nand Attention Mask", ha="center", va="center", fontsize=12, color="#8A8A8A")
    ax.text(11.7, 0.74, "Inputs", ha="center", va="center", fontsize=12, color="#9A9A9A")

    _arrow(ax, (11.7, 0.98), (11.7, 2.08), lw=1.6)
    _arrow(ax, (11.7, 2.31), (11.7, 3.16), lw=1.6)
    _arrow(ax, (11.7, 4.18), (11.7, 4.38), lw=1.6)
    _arrow(ax, (11.7, 4.82), (11.7, 5.0), lw=1.6)
    _arrow(ax, (11.7, 5.82), (11.7, 5.96), lw=1.6)
    _arrow(ax, (11.7, 6.40), (11.7, 6.56), lw=1.6)
    _arrow(ax, (11.7, 7.12), (11.7, 7.28), lw=1.6)
    _arrow(ax, (11.7, 7.76), (11.7, 7.88), lw=1.6)

    _arrow(ax, (9.78, 2.08), (10.18, 3.66), lw=1.4, rad=0.0)
    _arrow(ax, (13.15, 2.31), (13.15, 3.16), lw=1.4)
    _arrow(ax, (8.98, 2.08), (9.32, 2.08), lw=1.2)
    _arrow(ax, (13.38, 2.08), (13.67, 2.08), lw=1.2)

    _arrow(ax, (8.4, 5.67), (10.18, 4.00), lw=1.25, rad=-0.08)
    _arrow(ax, (8.4, 4.79), (10.18, 3.82), lw=1.25, rad=0.0)
    _arrow(ax, (8.4, 3.91), (10.18, 3.64), lw=1.25, rad=0.08)

    _skip(ax, 13.22, 3.74, 13.76, 4.38, 12.98, 4.60)
    _skip(ax, 13.22, 5.42, 13.82, 5.96, 12.98, 6.18)

    ax.text(13.90, 4.56, "skip", fontsize=9.5, color=COLORS["residual"], ha="left")
    ax.text(13.95, 6.14, "skip", fontsize=9.5, color=COLORS["residual"], ha="left")

    ax.set_xlim(0.2, 16.2)
    ax.set_ylim(0.2, 8.8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Render a dl-visuals-inspired diagram for src/models/bert.py.")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "bert_dl_visuals_style.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_bert_visual(output_path)
    print(f"[ok] saved -> {output_path}")


if __name__ == "__main__":
    main()
