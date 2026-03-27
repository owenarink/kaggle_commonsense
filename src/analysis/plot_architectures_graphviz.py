from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src.analysis.architecture_viz_utils import OUTPUT_ROOT, PALETTE, ModelVizSpec, ensure_dir, selected_specs


def _maybe_graphviz():
    try:
        from graphviz import Digraph
    except Exception as exc:  # pragma: no cover - dependency dependent
        raise RuntimeError(
            "graphviz package is required for architecture graphs. Install python graphviz and the Graphviz binary."
        ) from exc
    return Digraph


def _node_color(kind: str) -> str:
    return PALETTE.get(kind, "#CED4DA")


def _draw_box(ax, x, y, label, color, width=1.7):
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#1F2937",
        bbox={
            "boxstyle": "round,pad=0.45,rounding_size=0.12",
            "facecolor": color,
            "edgecolor": "#2B2D42",
            "linewidth": 1.5,
        },
    )


def _draw_arrow(ax, start, end, color="#6C757D"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=color,
            connectionstyle="arc3,rad=0.0",
        )
    )


def _render_matplotlib_architecture(spec: ModelVizSpec, output_dir: Path):
    is_attention = spec.key == "transformer_attentiontypes"
    fig = plt.figure(figsize=(14, 6.5 if is_attention else 4.8))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#FCFBF7")
    ax.set_facecolor("#FCFBF7")
    ax.set_title(f"{spec.title}\n{spec.summary}", fontsize=17, color="#1D3557", pad=16)

    if is_attention:
        top_y = 3.9
        middle_y = 2.35
        bottom_y = 0.95
        positions = []
        for idx, layer in enumerate(spec.layers[:4]):
            x = 1.7 + idx * 2.35
            positions.append((x, top_y, layer))
        for idx, layer in enumerate(spec.layers[4:7], start=4):
            x = 4.0 + (idx - 4) * 2.6
            positions.append((x, middle_y, layer))
        for idx, layer in enumerate(spec.layers[7:], start=7):
            x = 6.0 + (idx - 7) * 2.6
            positions.append((x, bottom_y, layer))
    else:
        y = 2.0
        positions = [(1.6 + idx * 2.15, y, layer) for idx, layer in enumerate(spec.layers)]

    coord_map = {}
    for idx, (x, y, layer) in enumerate(positions):
        label = layer.name if not layer.shape else f"{layer.name}\n{layer.shape}"
        coord_map[idx] = (x, y)
        _draw_box(ax, x, y, label, _node_color(layer.kind))

    if is_attention:
        top_idx = [0, 1, 2, 3]
        mid_idx = [4, 5, 6]
        bot_idx = [7, 8, 9]
        _draw_arrow(ax, (coord_map[0][0] + 0.8, coord_map[0][1]), (coord_map[1][0] - 0.8, coord_map[1][1]))
        _draw_arrow(ax, (coord_map[1][0] + 0.8, coord_map[1][1]), (coord_map[4][0] - 0.8, coord_map[4][1] + 0.1))
        _draw_arrow(ax, (coord_map[2][0], coord_map[2][1] - 0.45), (coord_map[4][0] + 0.4, coord_map[4][1] + 0.3))
        _draw_arrow(ax, (coord_map[2][0] + 0.7, coord_map[2][1] - 0.45), (coord_map[5][0] - 0.2, coord_map[5][1] + 0.3))
        _draw_arrow(ax, (coord_map[3][0] - 0.8, coord_map[3][1]), (coord_map[5][0] + 0.8, coord_map[5][1] + 0.1))
        _draw_arrow(ax, (coord_map[4][0] + 0.8, coord_map[4][1]), (coord_map[5][0] - 0.8, coord_map[5][1]))
        _draw_arrow(ax, (coord_map[5][0] + 0.8, coord_map[5][1]), (coord_map[6][0] - 0.8, coord_map[6][1]))
        _draw_arrow(ax, (coord_map[6][0], coord_map[6][1] - 0.45), (coord_map[7][0], coord_map[7][1] + 0.45))
        _draw_arrow(ax, (coord_map[7][0] + 0.8, coord_map[7][1]), (coord_map[8][0] - 0.8, coord_map[8][1]))
        _draw_arrow(ax, (coord_map[8][0] + 0.8, coord_map[8][1]), (coord_map[9][0] - 0.8, coord_map[9][1]))

        ax.text(
            11.5,
            2.5,
            "\n".join(f"- {point}" for point in spec.focus_points),
            ha="left",
            va="center",
            fontsize=9.5,
            color="#6B4F2B",
            bbox={
                "boxstyle": "round,pad=0.45,rounding_size=0.12",
                "facecolor": "#FEFAE0",
                "edgecolor": "#BC6C25",
                "linewidth": 1.3,
            },
        )
        ax.set_xlim(0.5, 16.5)
        ax.set_ylim(0.2, 4.7)
    else:
        for idx in range(len(spec.layers) - 1):
            x1, y1 = coord_map[idx]
            x2, y2 = coord_map[idx + 1]
            _draw_arrow(ax, (x1 + 0.72, y1), (x2 - 0.72, y2))
        ax.set_xlim(0.5, coord_map[len(spec.layers) - 1][0] + 1.8)
        ax.set_ylim(0.8, 3.1)

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{spec.key}_graphviz.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_architecture(spec: ModelVizSpec, output_dir: Path):
    try:
        Digraph = _maybe_graphviz()
        graph = Digraph(spec.key, format="png")
        graph.attr(rankdir="LR", splines="polyline", bgcolor="#FCFBF7", pad="0.4", nodesep="0.45", ranksep="0.75")
        graph.attr("node", shape="box", style="rounded,filled", color="#2B2D42", fontname="Helvetica", fontsize="12")
        graph.attr("edge", color="#6C757D", penwidth="1.6", arrowsize="0.8")
        graph.attr(label=f"{spec.title}\\n{spec.summary}", labelloc="t", fontsize="20", fontname="Helvetica-Bold")

        previous = None
        for index, layer in enumerate(spec.layers):
            node_id = f"n{index}"
            label = layer.name if not layer.shape else f"{layer.name}\\n{layer.shape}"
            if layer.note:
                label = f"{label}\\n{layer.note}"
            graph.node(node_id, label, fillcolor=_node_color(layer.kind))
            if previous is not None:
                graph.edge(previous, node_id)
            previous = node_id

        if spec.focus_points:
            with graph.subgraph(name=f"cluster_{spec.key}_notes") as subgraph:
                subgraph.attr(label="Key Notes", color="#BC6C25", style="rounded,dashed", fontname="Helvetica-Bold")
                for idx, point in enumerate(spec.focus_points):
                    note_id = f"note_{idx}"
                    subgraph.node(note_id, point, fillcolor="#FEFAE0", color="#BC6C25", fontsize="11")

        if spec.key == "transformer_attentiontypes":
            graph.node("a_q", "Token queries", fillcolor=PALETTE["attention"])
            graph.node("a_kv", "Token keys/values", fillcolor=PALETTE["attention"])
            graph.node("a_rel", "Relative position keys/queries", fillcolor=PALETTE["position"])
            graph.node("a_mix", "Disentangled score fusion", fillcolor=PALETTE["competition"])
            graph.edge("n2", "a_q")
            graph.edge("n2", "a_kv")
            graph.edge("n3", "a_rel")
            graph.edge("a_q", "a_mix")
            graph.edge("a_kv", "a_mix")
            graph.edge("a_rel", "a_mix")
            graph.edge("a_mix", "n4")

        graph.render(output_dir / f"{spec.key}_graphviz", cleanup=True)
    except Exception:
        _render_matplotlib_architecture(spec, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Render neat Graphviz architecture plots for each model.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model keys to render.")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT / "architectures" / "graphviz"))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    for spec in selected_specs(args.models):
        render_architecture(spec, output_dir)
        print(f"[ok] {spec.key} -> {output_dir / f'{spec.key}_graphviz.png'}")


if __name__ == "__main__":
    main()
