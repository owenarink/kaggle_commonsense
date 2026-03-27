from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src.analysis.architecture_viz_utils import OUTPUT_ROOT, ModelVizSpec, ensure_dir, selected_specs


def _tex_header(title: str) -> str:
    return rf"""\documentclass[tikz,border=8pt]{{standalone}}
\usepackage{{tikz}}
\usetikzlibrary{{positioning,calc}}
\begin{{document}}
\begin{{tikzpicture}}[
    layer/.style={{draw=black!80, thick, fill=blue!12, minimum width=2.8cm, minimum height=1.2cm, align=center}},
    biglayer/.style={{draw=black!80, thick, fill=teal!14, minimum width=3.2cm, minimum height=2.4cm, align=center}},
    arrow/.style={{-latex, thick}}
]
\node[align=center,font=\bfseries\Large] at (8,3.8) {{{title}}};
"""


def _pick_style(shape: str) -> str:
    return "biglayer" if shape.count("x") >= 2 or shape.count("X") >= 2 else "layer"


def render_tex(spec: ModelVizSpec, output_dir: Path):
    lines = [_tex_header(spec.title)]
    x = 0.0
    node_ids = []
    for idx, layer in enumerate(spec.layers):
        node_id = f"l{idx}"
        node_ids.append(node_id)
        shape = layer.shape or ""
        style = _pick_style(shape)
        label = layer.name if not shape else f"{layer.name}\\\\ {shape}"
        lines.append(rf"\node[{style}] ({node_id}) at ({x:.2f},0) {{{label}}};")
        x += 3.35

    for src, dst in zip(node_ids[:-1], node_ids[1:]):
        lines.append(rf"\draw[arrow] ({src}.east) -- ({dst}.west);")

    if spec.focus_points:
        note_text = r"\\ ".join(spec.focus_points)
        lines.append(rf"\node[draw=orange!70!black, fill=yellow!10, rounded corners, align=left, text width=5.4cm] at ({x + 2.2:.2f},0) {{{note_text}}};")

    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{document}")

    output_path = output_dir / f"{spec.key}_plotneuralnet.tex"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def render_preview_png(spec: ModelVizSpec, output_dir: Path):
    fig_width = max(11, len(spec.layers) * 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_title(f"{spec.title}\nPlotNeuralNet-style preview", fontsize=18, color="#1D3557", pad=18)

    x = 1.6
    positions = []
    for layer in spec.layers:
        shape = layer.shape or ""
        big = shape.count("x") >= 2 or shape.count("X") >= 2
        width = 1.7 if big else 1.45
        height = 1.85 if big else 1.0
        label = layer.name if not shape else f"{layer.name}\n{shape}"
        positions.append((x, width, height))
        ax.text(
            x,
            1.45,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.35,rounding_size=0.08",
                "facecolor": "#DCEAF7" if big else "#EAF7F5",
                "edgecolor": "#2B2D42",
                "linewidth": 1.4,
            },
        )
        x += 2.35

    for (x1, _, _), (x2, _, _) in zip(positions[:-1], positions[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.78, 1.45),
                (x2 - 0.78, 1.45),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.6,
                color="#4A5568",
            )
        )

    if spec.focus_points:
        ax.text(
            x + 1.2,
            1.45,
            "\n".join(f"- {point}" for point in spec.focus_points),
            ha="left",
            va="center",
            fontsize=9.5,
            color="#6B4F2B",
            bbox={
                "boxstyle": "round,pad=0.35,rounding_size=0.08",
                "facecolor": "#FEFAE0",
                "edgecolor": "#BC6C25",
                "linewidth": 1.2,
            },
        )

    ax.set_xlim(0.6, x + 5.5)
    ax.set_ylim(0.4, 2.6)
    ax.axis("off")
    fig.tight_layout()
    png_path = output_dir / f"{spec.key}_plotneuralnet.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return png_path


def compile_tex_outputs(tex_path: Path):
    pdflatex = shutil.which("pdflatex")
    pdftoppm = shutil.which("pdftoppm")
    if not pdflatex:
        return None, None, "pdflatex not found; wrote .tex only"

    workdir = tex_path.parent
    subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
    )
    pdf_path = tex_path.with_suffix(".pdf")

    png_path = None
    if pdftoppm and pdf_path.exists():
        png_base = tex_path.with_suffix("")
        subprocess.run(
            [pdftoppm, "-png", "-singlefile", str(pdf_path), str(png_base)],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        )
        candidate = png_base.with_suffix(".png")
        if candidate.exists():
            png_path = candidate

    return pdf_path if pdf_path.exists() else None, png_path, None


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PlotNeuralNet-style TikZ architecture files.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model keys to render.")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT / "architectures" / "plotneuralnet"))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    for spec in selected_specs(args.models):
        tex_path = render_tex(spec, output_dir)
        pdf_path, compiled_png_path, warning = compile_tex_outputs(tex_path)
        preview_png_path = render_preview_png(spec, output_dir)
        print(f"[ok] {spec.key} tex -> {tex_path}")
        if pdf_path is not None:
            print(f"[ok] {spec.key} pdf -> {pdf_path}")
        if compiled_png_path is not None:
            print(f"[ok] {spec.key} png -> {compiled_png_path}")
        if warning is not None:
            print(f"[warn] {spec.key} {warning}")
        print(f"[ok] {spec.key} preview -> {preview_png_path}")


if __name__ == "__main__":
    main()
