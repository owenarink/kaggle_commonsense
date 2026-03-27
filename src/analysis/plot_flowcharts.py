from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src.analysis.architecture_viz_utils import OUTPUT_ROOT, PALETTE, ensure_dir


def _maybe_graphviz():
    try:
        from graphviz import Digraph
    except Exception as exc:  # pragma: no cover - dependency dependent
        raise RuntimeError(
            "graphviz package is required for flowcharts. Install python graphviz and the Graphviz binary."
        ) from exc
    return Digraph


def _style_graph(graph, title: str):
    graph.attr(rankdir="LR", splines="spline", bgcolor="#FAF7F2", pad="0.35")
    graph.attr("node", shape="box", style="rounded,filled", color="#3D405B", fontname="Helvetica", fontsize="12")
    graph.attr("edge", color="#5C677D", penwidth="1.6", arrowsize="0.8")
    graph.attr(label=title, labelloc="t", fontsize="20", fontname="Helvetica-Bold", fontcolor="#1D3557")


def _draw_box(ax, xy, text, color):
    ax.text(
        xy[0],
        xy[1],
        text,
        ha="center",
        va="center",
        fontsize=11,
        color="#1D3557",
        bbox={
            "boxstyle": "round,pad=0.5,rounding_size=0.2",
            "facecolor": color,
            "edgecolor": "#3D405B",
            "linewidth": 1.5,
        },
    )


def _draw_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.6,
        color="#5C677D",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def _render_matplotlib_flowchart(output_path: Path, title: str, nodes: dict[str, tuple[float, float, str, str]], edges: list[tuple[str, str]], figsize=(14, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#FAF7F2")
    ax.set_facecolor("#FAF7F2")
    ax.set_title(title, fontsize=18, color="#1D3557", pad=18)

    for _, (x, y, text, color) in nodes.items():
        _draw_box(ax, (x, y), text, color)

    for src, dst in edges:
        x1, y1, _, _ = nodes[src]
        x2, y2, _, _ = nodes[dst]
        _draw_arrow(ax, (x1 + 0.55, y1), (x2 - 0.55, y2))

    ax.set_xlim(0, max(v[0] for v in nodes.values()) + 1)
    ax.set_ylim(0, max(v[1] for v in nodes.values()) + 1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _render_graphviz_or_matplotlib(graph_builder, output_dir: Path, stem: str, title: str, fallback_nodes, fallback_edges, figsize=(14, 4)):
    try:
        graph = graph_builder()
        graph.render(output_dir / stem, cleanup=True)
    except Exception:
        _render_matplotlib_flowchart(output_dir / f"{stem}.png", title, fallback_nodes, fallback_edges, figsize=figsize)


def overall_methodology_chart(output_dir: Path):
    def build():
        Digraph = _maybe_graphviz()
        graph = Digraph("overall_methodology", format="png")
        _style_graph(graph, "Overall Methodology")
        graph.node("data", "Raw CSV Data\ntrain_data + train_answers + test_data", fillcolor=PALETTE["input"])
        graph.node("prep", "Preprocess\nstrip text, merge labels,\nvalidate A/B/C schema", fillcolor=PALETTE["process"])
        graph.node("repr", "Representation Branches\nTF-IDF / BBPE / word IDs", fillcolor=PALETTE["tokenizer"])
        graph.node("models", "Model Families\nMLP, CNN, Transformer,\nCounterfactual variants", fillcolor=PALETTE["attention"])
        graph.node("train", "Training\nCrossEntropy or BCEWithLogits\n+ validation split", fillcolor=PALETTE["process"])
        graph.node("eval", "Evaluation\ncategory accuracy,\nconfusion, loss landscape", fillcolor=PALETTE["artifact"])
        graph.node("submit", "Submission / Analysis Outputs", fillcolor=PALETTE["output"])
        graph.edges([("data", "prep"), ("prep", "repr"), ("repr", "models"), ("models", "train"), ("train", "eval"), ("eval", "submit")])
        return graph

    nodes = {
        "data": (1, 1.5, "Raw CSV Data\ntrain_data + train_answers + test_data", PALETTE["input"]),
        "prep": (3.2, 1.5, "Preprocess\nstrip text, merge labels,\nvalidate A/B/C schema", PALETTE["process"]),
        "repr": (5.5, 1.5, "Representation Branches\nTF-IDF / BBPE / word IDs", PALETTE["tokenizer"]),
        "models": (7.8, 1.5, "Model Families\nMLP, CNN, Transformer,\nCounterfactual variants", PALETTE["attention"]),
        "train": (10.1, 1.5, "Training\nCrossEntropy or BCEWithLogits\n+ validation split", PALETTE["process"]),
        "eval": (12.4, 1.5, "Evaluation\ncategory accuracy,\nconfusion, loss landscape", PALETTE["artifact"]),
        "submit": (14.7, 1.5, "Submission / Analysis Outputs", PALETTE["output"]),
    }
    edges = [("data", "prep"), ("prep", "repr"), ("repr", "models"), ("models", "train"), ("train", "eval"), ("eval", "submit")]
    _render_graphviz_or_matplotlib(build, output_dir, "overall_methodology", "Overall Methodology", nodes, edges, figsize=(16, 3.6))


def data_processing_chart(output_dir: Path):
    def build():
        Digraph = _maybe_graphviz()
        graph = Digraph("data_processing_pipeline", format="png")
        _style_graph(graph, "Data Processing Pipeline")
        graph.node("csv", "Load CSV files", fillcolor=PALETTE["input"])
        graph.node("check", "Column validation\nid, FalseSent, OptionA/B/C", fillcolor=PALETTE["decision"])
        graph.node("clean", "Basic cleanup\nastype(str), strip()", fillcolor=PALETTE["process"])
        graph.node("merge", "Merge answers into train\nrename answer -> label", fillcolor=PALETTE["process"])
        graph.node("split", "Stratified split\ntrain / validation", fillcolor=PALETTE["artifact"])
        graph.node("pair", "Optional pairwise expansion\nFalseSent [SEP] Option", fillcolor=PALETTE["tokenizer"])
        graph.node("encode", "Encode branch\nTF-IDF, BBPE, or vocab IDs", fillcolor=PALETTE["tokenizer"])
        graph.node("tensors", "Tensor datasets / dataloaders", fillcolor=PALETTE["output"])
        graph.edges([("csv", "check"), ("check", "clean"), ("clean", "merge"), ("merge", "split"), ("split", "pair"), ("pair", "encode"), ("encode", "tensors")])
        return graph

    nodes = {
        "csv": (1, 1.5, "Load CSV files", PALETTE["input"]),
        "check": (3, 1.5, "Column validation\nid, FalseSent, OptionA/B/C", PALETTE["decision"]),
        "clean": (5, 1.5, "Basic cleanup\nastype(str), strip()", PALETTE["process"]),
        "merge": (7, 1.5, "Merge answers into train\nrename answer -> label", PALETTE["process"]),
        "split": (9, 1.5, "Stratified split\ntrain / validation", PALETTE["artifact"]),
        "pair": (11, 1.5, "Optional pairwise expansion\nFalseSent [SEP] Option", PALETTE["tokenizer"]),
        "encode": (13, 1.5, "Encode branch\nTF-IDF, BBPE, or vocab IDs", PALETTE["tokenizer"]),
        "tensors": (15, 1.5, "Tensor datasets / dataloaders", PALETTE["output"]),
    }
    edges = [("csv", "check"), ("check", "clean"), ("clean", "merge"), ("merge", "split"), ("split", "pair"), ("pair", "encode"), ("encode", "tensors")]
    _render_graphviz_or_matplotlib(build, output_dir, "data_processing_pipeline", "Data Processing Pipeline", nodes, edges, figsize=(16, 3.6))


def tokenization_chart(output_dir: Path):
    def build():
        Digraph = _maybe_graphviz()
        graph = Digraph("tokenization_pipeline", format="png")
        _style_graph(graph, "Tokenization And Encoding")
        graph.node("text", "False sentence + options", fillcolor=PALETTE["input"])
        graph.node("tfidf", "TF-IDF branch\nbinary unigram/bigram\nsparse vector", fillcolor=PALETTE["tokenizer"])
        graph.node("cnn", "CNN branch\nsimple_tokenize -> vocab -> pad IDs", fillcolor=PALETTE["tokenizer"])
        graph.node("bbpe", "Transformer branch\nByte-level BPE\nspecial tokens + max_len", fillcolor=PALETTE["tokenizer"])
        graph.node("group", "Grouped encoding\n[batch, 3, seq_len]", fillcolor=PALETTE["artifact"])
        graph.node("pair", "Pairwise encoding\n[batch, seq_len] + BCE labels", fillcolor=PALETTE["artifact"])
        graph.edge("text", "tfidf")
        graph.edge("text", "cnn")
        graph.edge("text", "bbpe")
        graph.edge("cnn", "pair")
        graph.edge("bbpe", "group")
        graph.edge("tfidf", "pair")
        return graph

    nodes = {
        "text": (4.5, 4.2, "False sentence + options", PALETTE["input"]),
        "tfidf": (1.7, 2.5, "TF-IDF branch\nbinary unigram/bigram\nsparse vector", PALETTE["tokenizer"]),
        "cnn": (4.5, 2.5, "CNN branch\nsimple_tokenize -> vocab -> pad IDs", PALETTE["tokenizer"]),
        "bbpe": (7.3, 2.5, "Transformer branch\nByte-level BPE\nspecial tokens + max_len", PALETTE["tokenizer"]),
        "pair": (2.7, 0.9, "Pairwise encoding\n[batch, seq_len] + BCE labels", PALETTE["artifact"]),
        "group": (6.3, 0.9, "Grouped encoding\n[batch, 3, seq_len]", PALETTE["artifact"]),
    }
    edges = [("text", "tfidf"), ("text", "cnn"), ("text", "bbpe"), ("cnn", "pair"), ("tfidf", "pair"), ("bbpe", "group")]
    _render_graphviz_or_matplotlib(build, output_dir, "tokenization_pipeline", "Tokenization And Encoding", nodes, edges, figsize=(10, 5.2))


def attentiontypes_chart(output_dir: Path):
    def build():
        Digraph = _maybe_graphviz()
        graph = Digraph("attentiontypes_methodology", format="png")
        _style_graph(graph, "Attentiontypes Methodology")
        graph.attr(rankdir="TB")
        graph.node("input", "Grouped BBPE token IDs\n[batch, 3, seq]", fillcolor=PALETTE["input"])
        graph.node("embed", "Token embeddings", fillcolor=PALETTE["embedding"])
        graph.node("rel", "Relative position embeddings", fillcolor=PALETTE["position"])
        graph.node("c2c", "content -> content", fillcolor=PALETTE["attention"])
        graph.node("c2p", "content -> position", fillcolor=PALETTE["attention"])
        graph.node("p2c", "position -> content", fillcolor=PALETTE["attention"])
        graph.node("merge", "Combine attention terms\nscale / softmax / value mix", fillcolor=PALETTE["attention"])
        graph.node("ffn", "Feed-forward + residual + LayerNorm\nrepeat x4", fillcolor=PALETTE["ffn"])
        graph.node("pool", "Mean pool per option", fillcolor=PALETTE["pool"])
        graph.node("head", "Linear scorer\n1 logit per option", fillcolor=PALETTE["output"])
        graph.node("loss", "Grouped cross-entropy\nchoose A/B/C", fillcolor=PALETTE["decision"])
        graph.edge("input", "embed")
        graph.edge("embed", "c2c")
        graph.edge("embed", "c2p")
        graph.edge("rel", "c2p")
        graph.edge("rel", "p2c")
        graph.edge("embed", "p2c")
        graph.edge("c2c", "merge")
        graph.edge("c2p", "merge")
        graph.edge("p2c", "merge")
        graph.edge("merge", "ffn")
        graph.edge("ffn", "pool")
        graph.edge("pool", "head")
        graph.edge("head", "loss")
        return graph

    nodes = {
        "input": (4.2, 7.0, "Grouped BBPE token IDs\n[batch, 3, seq]", PALETTE["input"]),
        "embed": (4.2, 5.8, "Token embeddings", PALETTE["embedding"]),
        "rel": (7.3, 5.8, "Relative position embeddings", PALETTE["position"]),
        "c2c": (1.6, 4.2, "content -> content", PALETTE["attention"]),
        "c2p": (4.2, 4.2, "content -> position", PALETTE["attention"]),
        "p2c": (6.8, 4.2, "position -> content", PALETTE["attention"]),
        "merge": (4.2, 2.8, "Combine attention terms\nscale / softmax / value mix", PALETTE["attention"]),
        "ffn": (4.2, 1.6, "Feed-forward + residual + LayerNorm\nrepeat x4", PALETTE["ffn"]),
        "pool": (4.2, 0.4, "Mean pool per option", PALETTE["pool"]),
    }
    extra_nodes = {
        "head": (7.0, 0.4, "Linear scorer\n1 logit per option", PALETTE["output"]),
        "loss": (9.7, 0.4, "Grouped cross-entropy\nchoose A/B/C", PALETTE["decision"]),
    }
    nodes.update(extra_nodes)
    edges = [
        ("input", "embed"),
        ("embed", "c2c"),
        ("embed", "c2p"),
        ("rel", "c2p"),
        ("rel", "p2c"),
        ("embed", "p2c"),
        ("c2c", "merge"),
        ("c2p", "merge"),
        ("p2c", "merge"),
        ("merge", "ffn"),
        ("ffn", "pool"),
        ("pool", "head"),
        ("head", "loss"),
    ]
    _render_graphviz_or_matplotlib(build, output_dir, "attentiontypes_methodology", "Attentiontypes Methodology", nodes, edges, figsize=(11, 8))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate methodology and preprocessing flowcharts.")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT / "flowcharts"), help="Directory for rendered flowcharts.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    overall_methodology_chart(output_dir)
    data_processing_chart(output_dir)
    tokenization_chart(output_dir)
    attentiontypes_chart(output_dir)
    print(f"Saved flowcharts to {output_dir}")


if __name__ == "__main__":
    main()
