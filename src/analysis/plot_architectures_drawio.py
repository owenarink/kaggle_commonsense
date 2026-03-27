from __future__ import annotations

import argparse
import html
import math
import re
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

from src.analysis.architecture_viz_utils import OUTPUT_ROOT, PALETTE, ModelVizSpec, ensure_dir, selected_specs


@dataclass(frozen=True)
class BlockSpec:
    block_id: str
    title: str
    tensor_shape: str
    kind: str
    x: float
    y: float
    width: float
    height: float
    depth: float
    short_label: str = ""


def _parse_shape_dims(shape: str) -> list[int]:
    dims = [int(x) for x in re.findall(r"\d+", shape)]
    return dims or [1]


def _scaled_block_dims(shape: str, kind: str) -> tuple[float, float, float]:
    dims = _parse_shape_dims(shape)
    if len(dims) == 1:
        spatial = dims[0]
        stacked = 1
        channels = 1
    elif len(dims) == 2:
        stacked = 1
        spatial = dims[0]
        channels = dims[1]
    else:
        stacked = dims[-3]
        spatial = dims[-2]
        channels = dims[-1]

    # Map tensor dimensions to visual axes like architecture papers:
    # front width ~= sequence/spatial extent, front height ~= stacked groups,
    # depth/thickness ~= feature/channel dimension.
    width = 34 + 16 * math.log2(max(spatial, 1) + 1)
    height = 24 + 7 * math.log2(max(stacked, 1) + 1)
    depth = 12 + 8.5 * math.log2(max(channels, 1) + 1)

    if kind in {"output", "decision"}:
        width *= 0.72
        depth *= 0.7
        height *= 0.82
    if kind == "pool":
        width *= 0.82
        height *= 0.72
        depth *= 0.7
    if kind == "input":
        depth *= 0.72
        height *= 0.86

    height = max(22.0, height)
    depth = max(6.0, depth)

    return width, height, depth


def _edge_style() -> str:
    return (
        "endArrow=block;endFill=1;html=1;rounded=0;curved=1;"
        "strokeColor=#6C757D;strokeWidth=2;shadow=1;"
    )


def _cube_style(kind: str) -> str:
    fill = PALETTE.get(kind, "#CED4DA")
    return (
        "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;shadow=0;"
        "rotation=90;"
        "fontSize=12;fontColor=#1F2937;strokeColor=#2B2D42;strokeWidth=1.5;"
        f"fillColor={fill};size=18;"
    )


def _make_root(diagram_name: str):
    mxfile = Element("mxfile", host="app.diagrams.net", version="24.7.17")
    diagram = SubElement(mxfile, "diagram", name=diagram_name, id=str(uuid.uuid4())[:12])
    model = SubElement(
        diagram,
        "mxGraphModel",
        dx="1600",
        dy="900",
        grid="1",
        gridSize="10",
        guides="1",
        tooltips="1",
        connect="1",
        arrows="1",
        fold="1",
        page="1",
        pageScale="1",
        pageWidth="1800",
        pageHeight="1000",
        math="0",
        shadow="0",
    )
    root = SubElement(model, "root")
    SubElement(root, "mxCell", id="0")
    SubElement(root, "mxCell", id="1", parent="0")
    return mxfile, root


def _add_vertex(root, block: BlockSpec):
    cell = SubElement(
        root,
        "mxCell",
        id=block.block_id,
        value="",
        style=_cube_style(block.kind),
        vertex="1",
        parent="1",
    )
    SubElement(
        cell,
        "mxGeometry",
        x=str(block.x),
        y=str(block.y),
        width=str(block.width),
        height=str(block.height),
        as_="geometry",
    )


def _add_label(root, label_id: str, block: BlockSpec):
    label = block.short_label or f"{block.title}\n{block.tensor_shape}"
    cell = SubElement(
        root,
        "mxCell",
        id=label_id,
        value=html.escape(label).replace("\n", "&#xa;"),
        style=(
            "text;html=1;whiteSpace=wrap;align=center;verticalAlign=top;"
            "fontSize=12;fontColor=#1F2937;strokeColor=none;fillColor=none;"
        ),
        vertex="1",
        parent="1",
    )
    SubElement(
        cell,
        "mxGeometry",
        x=str(block.x - 12),
        y=str(block.y + block.height + block.depth * 0.55 + 14),
        width=str(max(85.0, block.width + block.depth + 10)),
        height="42",
        as_="geometry",
    )


def _add_text_cell(root, cell_id: str, text: str, x: float, y: float, width: float, height: float, align: str = "left"):
    cell = SubElement(
        root,
        "mxCell",
        id=cell_id,
        value=html.escape(text).replace("\n", "&#xa;"),
        style=(
            f"text;html=1;whiteSpace=wrap;align={align};verticalAlign=middle;"
            "fontSize=12;fontColor=#1F2937;strokeColor=none;fillColor=none;"
        ),
        vertex="1",
        parent="1",
    )
    SubElement(
        cell,
        "mxGeometry",
        x=str(x),
        y=str(y),
        width=str(width),
        height=str(height),
        as_="geometry",
    )


def _block_visual_width(block: BlockSpec) -> float:
    return max(90.0, block.height + block.depth + 12)


def _block_top_label_frame(block: BlockSpec) -> tuple[float, float, float, float]:
    width = max(150.0, min(300.0, _block_visual_width(block) + 105))
    x = block.x + (block.height + block.depth) * 0.5 - width * 0.5
    y = block.y - block.depth * 0.55 - 84
    return x, y, width, 30.0


def _block_shape_frame(block: BlockSpec) -> tuple[float, float, float, float]:
    width = max(84.0, min(170.0, _block_visual_width(block) + 20))
    x = block.x + (block.height + block.depth) * 0.5 - width * 0.5
    y = block.y + block.width + 16
    return x, y, width, 20.0


def _wrapped_title(block: BlockSpec, width: float) -> str:
    max_chars = max(12, min(34, int((width - 34) / 7.2)))
    return "\n".join(textwrap.wrap(block.title, width=max_chars, break_long_words=False))


def _label_box(block: BlockSpec) -> tuple[float, float, float]:
    _, _, width, _ = _block_top_label_frame(block)
    wrapped = _wrapped_title(block, width)
    lines = max(1, wrapped.count("\n") + 1)
    text_width = min(width, max(120.0, max(len(line) for line in wrapped.split("\n")) * 7.2 + 34))
    height = 18.0 * lines + 8.0
    return text_width, height, lines


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _top_label_positions(blocks: list[BlockSpec], spec: ModelVizSpec) -> dict[str, tuple[float, float, float, float]]:
    positions: dict[str, tuple[float, float, float, float]] = {}
    lanes: list[list[tuple[float, float]]] = []
    attention_ids = {"b3", "b4", "b5"} if spec.key == "transformer_attentiontypes" else set()

    for block in blocks:
        if block.block_id in attention_ids:
            continue
        x, y, width, _ = _block_top_label_frame(block)
        text_width, text_height, _ = _label_box(block)
        x = x + (width - text_width) * 0.5
        span = (x, x + text_width)
        lane_index = 0
        while lane_index < len(lanes) and any(_overlaps(span, other) for other in lanes[lane_index]):
            lane_index += 1
        if lane_index == len(lanes):
            lanes.append([])
        lanes[lane_index].append(span)
        y_pos = y - lane_index * (text_height + 12)
        if block.block_id == "b9":
            y_pos -= 30
        positions[block.block_id] = (x, y_pos, text_width, text_height)

    if attention_ids:
        stack_x = 500
        stack_y = 60
        stack_width = 430
        current_y = stack_y
        for block_id in ("b3", "b4", "b5"):
            block = next(block for block in blocks if block.block_id == block_id)
            wrapped = _wrapped_title(block, stack_width)
            lines = max(1, wrapped.count("\n") + 1)
            stack_height = 18.0 * lines + 6.0
            positions[block_id] = (stack_x, current_y, stack_width, stack_height)
            current_y += stack_height + 8.0

    return positions


def _title_marker_position(block: BlockSpec, label_frame: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, _, height = label_frame
    marker_y = y + height * 0.5
    if block.block_id in {"b3", "b4", "b5"}:
        marker_y -= 4
    return x + 6, marker_y


def _add_round_marker(root, marker_id: str, x: float, y: float, color: str, text: str = ""):
    cell = SubElement(
        root,
        "mxCell",
        id=marker_id,
        value=html.escape(text),
        style=(
            "ellipse;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;"
            "fontSize=10;fontStyle=1;fontColor=#FFFFFF;strokeColor=#2B2D42;strokeWidth=1.2;"
            f"fillColor={color};aspect=fixed;"
        ),
        vertex="1",
        parent="1",
    )
    SubElement(
        cell,
        "mxGeometry",
        x=str(x),
        y=str(y),
        width="16",
        height="16",
        as_="geometry",
    )


def _add_edge(root, edge_id: str, source: str, target: str):
    cell = SubElement(
        root,
        "mxCell",
        id=edge_id,
        style=_edge_style(),
        edge="1",
        parent="1",
        source=source,
        target=target,
    )
    SubElement(cell, "mxGeometry", relative="1", as_="geometry")


def _preview_entry_point(block: BlockSpec) -> tuple[float, float]:
    return (block.x - 6, block.y + block.width * 0.5)


def _preview_exit_point(block: BlockSpec) -> tuple[float, float]:
    return (block.x + block.height + block.depth + 6, block.y + block.width * 0.5 - block.depth * 0.12)


def _shape_catalog(spec: ModelVizSpec) -> list[BlockSpec]:
    if spec.key == "mlp_tfidf":
        return [
            BlockSpec("b0", "TF-IDF Input", "[1 x 20000]", "input", 60, 180, *_scaled_block_dims("[1 x 20000]", "input")),
            BlockSpec("b1", "Dense 64", "[1 x 64]", "dense", 250, 210, *_scaled_block_dims("[1 x 64]", "dense")),
            BlockSpec("b2", "3-Class Head", "[1 x 3]", "output", 385, 250, *_scaled_block_dims("[1 x 3]", "output")),
        ]
    if spec.key == "mlp_pairwise":
        return [
            BlockSpec("b0", "Pair TF-IDF", "[1 x 20000]", "input", 60, 180, *_scaled_block_dims("[1 x 20000]", "input")),
            BlockSpec("b1", "Dense 64", "[1 x 64]", "dense", 250, 210, *_scaled_block_dims("[1 x 64]", "dense")),
            BlockSpec("b2", "Binary Score", "[1 x 1]", "output", 385, 255, *_scaled_block_dims("[1 x 1]", "output")),
        ]
    if spec.key == "cnn_pairwise":
        return [
            BlockSpec("b0", "Token IDs", "[128]", "input", 45, 250, *_scaled_block_dims("[1 x 128 x 1]", "input")),
            BlockSpec("b1", "Embedding", "[128 x 128]", "embedding", 135, 175, *_scaled_block_dims("[1 x 128 x 128]", "embedding")),
            BlockSpec("b2", "Res Stage 64", "[128 x 64]", "cnn", 320, 205, *_scaled_block_dims("[1 x 128 x 64]", "cnn")),
            BlockSpec("b3", "Res Stage 128", "[64 x 128]", "residual", 490, 185, *_scaled_block_dims("[1 x 64 x 128]", "residual")),
            BlockSpec("b4", "Res Stage 256", "[32 x 256]", "residual", 675, 160, *_scaled_block_dims("[1 x 32 x 256]", "residual")),
            BlockSpec("b5", "Global Pool", "[256]", "pool", 880, 250, *_scaled_block_dims("[1 x 256]", "pool")),
            BlockSpec("b6", "Binary Head", "[1]", "output", 995, 260, *_scaled_block_dims("[1 x 1]", "output")),
        ]
    if spec.key == "transformer_attentiontypes":
        return [
            BlockSpec("b0", "Grouped BBPE", "[3 x 128]", "input", 40, 255, *_scaled_block_dims("[3 x 128 x 1]", "input"), short_label=""),
            BlockSpec("b1", "Token Embedding Layer", "[3 x 128 x 256]", "embedding", 155, 255, *_scaled_block_dims("[3 x 128 x 256]", "embedding"), short_label=""),
            BlockSpec("b2", "Relative Positional Embedding Layer", "[128 x 128 x 256]", "position", 330, 255, *_scaled_block_dims("[128 x 128 x 256]", "position"), short_label=""),
            BlockSpec("b3", "Content-to-Content Attention Scores", "[3 x 128 x 256]", "attention", 565, 255, *_scaled_block_dims("[3 x 128 x 256]", "attention"), short_label="1"),
            BlockSpec("b4", "Content-to-Position Attention Scores", "[3 x 128 x 256]", "attention", 705, 255, *_scaled_block_dims("[3 x 128 x 256]", "attention"), short_label="2"),
            BlockSpec("b5", "Position-to-Content Attention Scores", "[3 x 128 x 256]", "attention", 845, 255, *_scaled_block_dims("[3 x 128 x 256]", "attention"), short_label="3"),
            BlockSpec("b6", "Disentangled Multi-Head Self-Attention", "[3 x 128 x 256]", "competition", 1005, 255, *_scaled_block_dims("[3 x 128 x 256]", "competition"), short_label=""),
            BlockSpec("b7", "Position-wise Feed-Forward + Residual + LayerNorm", "[3 x 128 x 256]", "ffn", 1185, 255, *_scaled_block_dims("[3 x 128 x 256]", "ffn"), short_label=""),
            BlockSpec("b8", "Mean Pooling Layer", "[3 x 256]", "pool", 1375, 255, *_scaled_block_dims("[3 x 256]", "pool"), short_label=""),
            BlockSpec("b9", "Linear Classification Head", "[3 x 1]", "output", 1490, 255, *_scaled_block_dims("[3 x 1]", "output"), short_label=""),
            BlockSpec("b10", "Grouped Cross-Entropy Loss", "[3 classes]", "decision", 1575, 255, *_scaled_block_dims("[1 x 3]", "decision"), short_label=""),
        ]
    if spec.key == "transformer_bbpe":
        return [
            BlockSpec("b0", "Grouped BBPE Input", "[3 x 128]", "input", 60, 240, *_scaled_block_dims("[3 x 128 x 1]", "input")),
            BlockSpec("b1", "Token Embedding Layer", "[3 x 128 x 256]", "embedding", 235, 240, *_scaled_block_dims("[3 x 128 x 256]", "embedding")),
            BlockSpec("b2", "Positional Encoding", "[3 x 128 x 256]", "position", 470, 240, *_scaled_block_dims("[3 x 128 x 256]", "position")),
            BlockSpec("b3", "Transformer Encoder x4", "[3 x 128 x 256]", "attention", 725, 240, *_scaled_block_dims("[3 x 128 x 256]", "attention")),
            BlockSpec("b4", "Mean Pooling", "[3 x 256]", "pool", 1025, 240, *_scaled_block_dims("[3 x 256]", "pool")),
            BlockSpec("b5", "Linear Score Head", "[3 x 1]", "output", 1190, 240, *_scaled_block_dims("[3 x 1]", "output")),
            BlockSpec("b6", "Grouped Cross-Entropy", "[3 classes]", "decision", 1315, 240, *_scaled_block_dims("[1 x 3]", "decision")),
        ]
    if spec.key in {"transformer_attention_relsimple", "transformer_attention_sepdist"}:
        special = "SEP Dist" if spec.key.endswith("sepdist") else "RelSimple"
        return [
            BlockSpec("b0", "Grouped BBPE", "[3 x 128]", "input", 45, 265, *_scaled_block_dims("[3 x 128 x 1]", "input")),
            BlockSpec("b1", "Embeddings", "[3 x 128 x 256]", "embedding", 135, 155, *_scaled_block_dims("[3 x 128 x 256]", "embedding")),
            BlockSpec("b2", special, "[3 x 128 x 256]", "position", 360, 175, *_scaled_block_dims("[3 x 128 x 256]", "position")),
            BlockSpec("b3", "Flexible Attn x4", "[3 x 128 x 256]", "attention", 590, 150, *_scaled_block_dims("[3 x 128 x 256]", "attention")),
            BlockSpec("b4", "Mean Pool", "[3 x 256]", "pool", 840, 285, *_scaled_block_dims("[3 x 256]", "pool")),
            BlockSpec("b5", "Score Head", "[3 x 1]", "output", 970, 300, *_scaled_block_dims("[3 x 1]", "output")),
        ]
    if spec.key == "transformer_bertcounterfact":
        return [
            BlockSpec("b0", "Grouped BBPE", "[3 x 128]", "input", 40, 265, *_scaled_block_dims("[3 x 128 x 1]", "input")),
            BlockSpec("b1", "Encoder", "[3 x 128 x 256]", "attention", 130, 150, *_scaled_block_dims("[3 x 128 x 256]", "attention")),
            BlockSpec("b2", "Repair Module", "[3 x 256]", "competition", 370, 200, *_scaled_block_dims("[3 x 256]", "competition")),
            BlockSpec("b3", "Fuse", "[3 x 256]", "dense", 535, 220, *_scaled_block_dims("[3 x 256]", "dense")),
            BlockSpec("b4", "Score Head", "[3 x 1]", "output", 665, 300, *_scaled_block_dims("[3 x 1]", "output")),
        ]
    if spec.key == "transformer_bertcounter_cross_option":
        return [
            BlockSpec("b0", "Grouped BBPE", "[3 x 128]", "input", 35, 265, *_scaled_block_dims("[3 x 128 x 1]", "input")),
            BlockSpec("b1", "Shared Encoder", "[3 x 128 x 256]", "attention", 120, 150, *_scaled_block_dims("[3 x 128 x 256]", "attention")),
            BlockSpec("b2", "Repair Vectors", "[3 x 256]", "competition", 360, 200, *_scaled_block_dims("[3 x 256]", "competition")),
            BlockSpec("b3", "Cross-Option\nCompetition", "[3 x 256]", "competition", 530, 190, *_scaled_block_dims("[3 x 256]", "competition")),
            BlockSpec("b4", "Final Projection", "[3 x 256]", "dense", 725, 220, *_scaled_block_dims("[3 x 256]", "dense")),
            BlockSpec("b5", "Grouped Scores", "[3]", "output", 875, 300, *_scaled_block_dims("[1 x 3]", "output")),
        ]
    if spec.key == "transformer_bertcounter_lec":
        return [
            BlockSpec("b0", "Grouped BBPE", "[3 x 128]", "input", 35, 265, *_scaled_block_dims("[3 x 128 x 1]", "input")),
            BlockSpec("b1", "Shared Encoder", "[3 x 128 x 256]", "attention", 120, 150, *_scaled_block_dims("[3 x 128 x 256]", "attention")),
            BlockSpec("b2", "Latent Edit", "[3 x 256]", "competition", 360, 200, *_scaled_block_dims("[3 x 256]", "competition")),
            BlockSpec("b3", "Option Competition", "[3 x 256]", "competition", 535, 190, *_scaled_block_dims("[3 x 256]", "competition")),
            BlockSpec("b4", "Final Projection", "[3 x 256]", "dense", 725, 220, *_scaled_block_dims("[3 x 256]", "dense")),
            BlockSpec("b5", "Grouped Scores", "[3]", "output", 875, 300, *_scaled_block_dims("[1 x 3]", "output")),
        ]
    return [
        BlockSpec("b0", spec.layers[0].name, spec.layers[0].shape or "[1 x 32]", spec.layers[0].kind, 45, 240, *_scaled_block_dims(spec.layers[0].shape or "[1 x 32]", spec.layers[0].kind)),
        BlockSpec("b1", spec.layers[-1].name, spec.layers[-1].shape or "[1 x 3]", spec.layers[-1].kind, 180, 245, *_scaled_block_dims(spec.layers[-1].shape or "[1 x 3]", spec.layers[-1].kind)),
    ]


def _edges_for_spec(spec: ModelVizSpec, blocks: list[BlockSpec]) -> list[tuple[str, str, str]]:
    if spec.key == "transformer_attentiontypes":
        return [(f"e{idx}", blocks[idx].block_id, blocks[idx + 1].block_id) for idx in range(len(blocks) - 1)]
    if spec.key == "transformer_bbpe":
        return [(f"e{idx}", blocks[idx].block_id, blocks[idx + 1].block_id) for idx in range(len(blocks) - 1)]
    return []


def _note_text(spec: ModelVizSpec) -> str:
    if spec.focus_points:
        return "\n".join(f"- {point}" for point in spec.focus_points)
    return ""


def render_drawio_file(spec: ModelVizSpec, output_dir: Path):
    mxfile, root = _make_root(spec.title)
    blocks = _shape_catalog(spec)
    edges = _edges_for_spec(spec, blocks)
    label_positions = _top_label_positions(blocks, spec)

    for block in blocks:
        _add_vertex(root, block)
    for edge_id, source, target in edges:
        _add_edge(root, edge_id, source, target)

    for block in blocks:
        label_x, label_y, label_w, label_h = label_positions[block.block_id]
        marker_x, marker_y = _title_marker_position(block, (label_x, label_y, label_w, label_h))
        _add_round_marker(root, f"{block.block_id}_title_marker", marker_x, marker_y - 8, PALETTE.get(block.kind, "#CED4DA"), block.short_label if block.block_id in {"b3", "b4", "b5"} else "")
        _add_text_cell(root, f"{block.block_id}_title", _wrapped_title(block, label_w - 28), label_x + 28, label_y, label_w - 28, label_h, align="left")
        shape_x, shape_y, shape_w, shape_h = _block_shape_frame(block)
        _add_text_cell(root, f"{block.block_id}_shape", block.tensor_shape, shape_x, shape_y, shape_w, shape_h, align="center")

    output_path = output_dir / f"{spec.key}_drawio.drawio"
    ElementTree(mxfile).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _darken(hex_color: str, factor: float = 0.82) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(max(0.0, min(1.0, c * factor)) for c in rgb)


def _lighten(hex_color: str, factor: float = 1.12) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(max(0.0, min(1.0, c * factor)) for c in rgb)


def _draw_block(ax, block: BlockSpec):
    fill = PALETTE.get(block.kind, "#CED4DA")
    # Quarter-turn the slab so the thin face is oriented differently.
    front = [
        (block.x, block.y),
        (block.x + block.height, block.y),
        (block.x + block.height, block.y + block.width),
        (block.x, block.y + block.width),
    ]
    side = [
        (block.x + block.height, block.y),
        (block.x + block.height + block.depth, block.y - block.depth * 0.55),
        (block.x + block.height + block.depth, block.y + block.width - block.depth * 0.55),
        (block.x + block.height, block.y + block.width),
    ]
    top = [
        (block.x, block.y),
        (block.x + block.depth, block.y - block.depth * 0.55),
        (block.x + block.height + block.depth, block.y - block.depth * 0.55),
        (block.x + block.height, block.y),
    ]
    ax.add_patch(Polygon(front, closed=True, facecolor=fill, edgecolor="#2B2D42", linewidth=1.5))
    ax.add_patch(Polygon(side, closed=True, facecolor=_darken(fill), edgecolor="#2B2D42", linewidth=1.3))
    ax.add_patch(Polygon(top, closed=True, facecolor=_lighten(fill), edgecolor="#2B2D42", linewidth=1.3))
    if block.short_label:
        ax.text(
            block.x + (block.height + block.depth) / 2.0,
            block.y + block.width / 2.0,
            block.short_label,
            ha="center",
            va="center",
            fontsize=12,
            color="white" if block.kind in {"attention", "competition"} else "#132A3E",
            fontweight="bold",
        )


def _draw_edge(ax, src: BlockSpec, dst: BlockSpec):
    start = _preview_exit_point(src)
    end = _preview_entry_point(dst)
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="Simple,head_length=10,head_width=10,tail_width=1.7",
        connectionstyle="arc3,rad=0.0",
        color="#6C757D",
        linewidth=1.6,
        alpha=0.95,
        zorder=2,
    )
    ax.add_patch(patch)


def _draw_preview(spec: ModelVizSpec, output_dir: Path):
    blocks = _shape_catalog(spec)
    edges = _edges_for_spec(spec, blocks)
    label_positions = _top_label_positions(blocks, spec)
    fig = plt.figure(figsize=(15, 7 if spec.key == "transformer_attentiontypes" else 5))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor("#F8F9FB")
    ax.set_facecolor("#F8F9FB")
    ax.set_title(f"{spec.title}\n3D architecture diagram with tensor shapes", fontsize=18, color="#1D3557", pad=18)

    block_map = {block.block_id: block for block in blocks}
    for _, source, target in edges:
        _draw_edge(ax, block_map[source], block_map[target])
    for block in blocks:
        _draw_block(ax, block)
    for block in blocks:
        label_x, label_y, label_w, label_h = label_positions[block.block_id]
        marker_x, marker_y = _title_marker_position(block, (label_x, label_y, label_w, label_h))
        ax.scatter(
            [marker_x + 8],
            [marker_y],
            s=105,
            c=[PALETTE.get(block.kind, "#CED4DA")],
            edgecolors="#2B2D42",
            linewidths=1.0,
            zorder=6,
        )
        if block.block_id in {"b3", "b4", "b5"}:
            ax.text(marker_x + 8, marker_y, block.short_label, ha="center", va="center", fontsize=8.3, color="white", fontweight="bold", zorder=7)
        ax.text(
            label_x + 28,
            label_y + label_h / 2.0,
            _wrapped_title(block, label_w - 28),
            ha="left",
            va="center",
            fontsize=9.9 if spec.key == "transformer_attentiontypes" else 9.3,
            color="#132A3E",
            zorder=6,
        )
        shape_x, shape_y, shape_w, shape_h = _block_shape_frame(block)
        ax.text(
            shape_x + shape_w / 2.0,
            shape_y + shape_h / 2.0,
            block.tensor_shape,
            ha="center",
            va="center",
            fontsize=9.4,
            color="#516070",
            zorder=6,
        )

    if spec.key == "transformer_bbpe":
        legend_items = [
            ("Dark block = input/output stage", PALETTE["input"]),
            ("Cool colors = encoder internals", PALETTE["attention"]),
            ("Arrows = data flow between stages", "#6C757D"),
            ("Tensor shapes shown below each block", "#ADB5BD"),
        ]
        lx, ly = 70, 760
        for idx, (text, color) in enumerate(legend_items):
            yy = ly + idx * 34
            ax.scatter([lx], [yy], s=120, c=[color], edgecolors="#2B2D42", linewidths=1.0, zorder=6)
            ax.text(lx + 22, yy, text, ha="left", va="center", fontsize=10.0, color="#384657", zorder=6)

        ax.text(
            860,
            770,
            "Encoder block repeats 4 times with 8 heads and model dimension 256.",
            ha="left",
            va="center",
            fontsize=10.2,
            color="#516070",
            zorder=6,
        )

    ax.set_xlim(0, 1780 if spec.key == "transformer_attentiontypes" else 1680 if spec.key == "transformer_bbpe" else 1850)
    ax.set_ylim(920 if spec.key == "transformer_bbpe" else 860, 0)
    ax.axis("off")
    fig.tight_layout()
    preview_path = output_dir / f"{spec.key}_drawio_preview.png"
    fig.savefig(preview_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return preview_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate diagrams.net-style 3D architecture diagrams.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model keys to render.")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT / "architectures" / "drawio"))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    for spec in selected_specs(args.models):
        drawio_path = render_drawio_file(spec, output_dir)
        preview_path = _draw_preview(spec, output_dir)
        print(f"[ok] {spec.key} -> {drawio_path}")
        print(f"[ok] {spec.key} preview -> {preview_path}")


if __name__ == "__main__":
    main()
