from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "plots"


@dataclass(frozen=True)
class LayerSpec:
    name: str
    kind: str
    shape: str = ""
    note: str = ""


@dataclass(frozen=True)
class ModelVizSpec:
    key: str
    title: str
    family: str
    summary: str
    layers: tuple[LayerSpec, ...]
    focus_points: tuple[str, ...] = ()


PALETTE = {
    "input": "#F6BD60",
    "tokenizer": "#84A59D",
    "embedding": "#6D597A",
    "attention": "#457B9D",
    "position": "#4D908E",
    "ffn": "#90BE6D",
    "norm": "#A8DADC",
    "pool": "#F28482",
    "head": "#E76F51",
    "residual": "#577590",
    "competition": "#BC6C25",
    "cnn": "#355070",
    "dense": "#B56576",
    "output": "#E63946",
    "process": "#5C677D",
    "decision": "#9A031E",
    "artifact": "#2A9D8F",
}


def _specs() -> tuple[ModelVizSpec, ...]:
    return (
        ModelVizSpec(
            key="mlp_tfidf",
            title="MLP TF-IDF",
            family="baseline",
            summary="Sparse TF-IDF baseline with a single hidden dense layer.",
            layers=(
                LayerSpec("Input", "input", "[batch, vocab]"),
                LayerSpec("TF-IDF Features", "tokenizer", "binary unigram+bigram"),
                LayerSpec("Dense 64", "dense", "ReLU"),
                LayerSpec("Classifier", "output", "3 logits"),
            ),
        ),
        ModelVizSpec(
            key="mlp_pairwise",
            title="MLP Pairwise",
            family="baseline",
            summary="Pairwise scorer on TF-IDF sentence-option concatenations.",
            layers=(
                LayerSpec("Input", "input", "[batch, vocab]"),
                LayerSpec("TF-IDF Pair Features", "tokenizer", "FalseSent [SEP] Option"),
                LayerSpec("Dense 64", "dense", "ReLU"),
                LayerSpec("Binary Score Head", "output", "1 logit"),
            ),
        ),
        ModelVizSpec(
            key="cnn_pairwise",
            title="Text ResNet CNN Pairwise",
            family="cnn",
            summary="Embedding plus residual 1D convolution stack for pairwise scoring.",
            layers=(
                LayerSpec("Token IDs", "input", "[batch, seq]"),
                LayerSpec("Embedding", "embedding", "128-d"),
                LayerSpec("Input Conv", "cnn", "Conv1D + BN + ReLU"),
                LayerSpec("Residual Stage 1", "residual", "2 blocks / 64 ch"),
                LayerSpec("Residual Stage 2", "residual", "2 blocks / 128 ch"),
                LayerSpec("Residual Stage 3", "residual", "2 blocks / 256 ch"),
                LayerSpec("Adaptive Max Pool", "pool", "global"),
                LayerSpec("Linear Head", "output", "1 logit"),
            ),
        ),
        ModelVizSpec(
            key="transformer_bbpe",
            title="Transformer BBPE",
            family="transformer",
            summary="Grouped transformer scorer over three options using BBPE tokenization.",
            layers=(
                LayerSpec("Grouped Token IDs", "input", "[batch, 3, seq]"),
                LayerSpec("BBPE Tokenizer", "tokenizer", "12k vocab"),
                LayerSpec("Embedding", "embedding", "256-d"),
                LayerSpec("Absolute Positional Encoding", "position", "sin/cos"),
                LayerSpec("Transformer Encoder x4", "attention", "8 heads"),
                LayerSpec("Mean Pool", "pool", "per option"),
                LayerSpec("Score Head", "output", "1 logit / option"),
                LayerSpec("Cross-Entropy over 3", "decision", "grouped choice"),
            ),
        ),
        ModelVizSpec(
            key="transformer_attentiontypes",
            title="Attention Types / DeBERTa-like",
            family="attentiontypes",
            summary="Most important model: disentangled attention combines content-content, content-position, and position-content terms.",
            layers=(
                LayerSpec("Grouped Token IDs", "input", "[batch, 3, seq]"),
                LayerSpec("BBPE Tokenizer", "tokenizer", "task-specific tokenizer"),
                LayerSpec("Embedding", "embedding", "256-d"),
                LayerSpec("Relative Position Embeddings", "position", "[2*max_len-1, 256]"),
                LayerSpec("Disentangled Multi-Head Self-Attention x4", "attention", "content-to-content + content-to-position + position-to-content"),
                LayerSpec("Position-wise Feed-Forward Network x4", "ffn", "4x expansion"),
                LayerSpec("Residual Connection + Layer Normalization", "norm", "per encoder block"),
                LayerSpec("Mean Pooling", "pool", "per option"),
                LayerSpec("Linear Classification Head", "output", "1 logit / option"),
                LayerSpec("Grouped Cross-Entropy Loss", "decision", "A/B/C choice"),
            ),
            focus_points=(
                "Relative position contributes directly inside self-attention rather than only through added positional embeddings.",
                "Each option is scored by the same encoder, then compared through grouped cross-entropy over the three candidates.",
                "This is the clearest transformer variant to highlight in the methodology diagrams.",
            ),
        ),
        ModelVizSpec(
            key="transformer_attention_relsimple",
            title="Attention RelSimple",
            family="transformer",
            summary="Simplified disentangled attention using content-to-position terms only.",
            layers=(
                LayerSpec("Grouped Token IDs", "input", "[batch, 3, seq]"),
                LayerSpec("BBPE Tokenizer", "tokenizer", "shared grouped encoding"),
                LayerSpec("Embedding", "embedding", "256-d"),
                LayerSpec("Relative Attention x4", "attention", "c2c + c2p"),
                LayerSpec("FFN + Norm", "ffn", "residual block"),
                LayerSpec("Mean Pool", "pool", "per option"),
                LayerSpec("Score Head", "output", "1 logit / option"),
            ),
        ),
        ModelVizSpec(
            key="transformer_attention_sepdist",
            title="Attention SEP Distance",
            family="transformer",
            summary="Flexible attention augments relative attention with distance-from-SEP bias.",
            layers=(
                LayerSpec("Grouped Token IDs", "input", "[batch, 3, seq]"),
                LayerSpec("BBPE Tokenizer", "tokenizer", "shared grouped encoding"),
                LayerSpec("Embedding", "embedding", "256-d"),
                LayerSpec("SEP Distance Features", "position", "distance to split token"),
                LayerSpec("Flexible Attention x4", "attention", "c2c + c2p + p2c + sepdist"),
                LayerSpec("FFN + Norm", "ffn", "residual block"),
                LayerSpec("Mean Pool", "pool", "per option"),
                LayerSpec("Score Head", "output", "1 logit / option"),
            ),
        ),
        ModelVizSpec(
            key="transformer_bertcounterfact",
            title="CounterFact Transformer",
            family="counterfactual",
            summary="DeBERTa-like encoder plus repair attention over false-sentence and option segments.",
            layers=(
                LayerSpec("Grouped Token IDs", "input", "[batch, 3, seq]"),
                LayerSpec("BBPE Tokenizer", "tokenizer", "false sentence + option"),
                LayerSpec("Embedding", "embedding", "256-d"),
                LayerSpec("Disentangled Encoder x4", "attention", "contextualization"),
                LayerSpec("Counterfactual Repair Module", "competition", "anomaly/support/conflict/repair"),
                LayerSpec("Fuse Pooled + Repair", "dense", "project + norm"),
                LayerSpec("Score Head", "output", "1 logit / option"),
            ),
        ),
        ModelVizSpec(
            key="transformer_bertcounter_cross_option",
            title="CounterFact Cross Option",
            family="counterfactual",
            summary="Adds explicit cross-option competition on top of counterfactual repair features.",
            layers=(
                LayerSpec("Grouped Token IDs", "input", "[batch, 3, seq]"),
                LayerSpec("Shared Encoder", "attention", "DeBERTa-like x4"),
                LayerSpec("Repair Module", "competition", "option-local repair vectors"),
                LayerSpec("Cross-Option Competition", "competition", "rival-aware comparison"),
                LayerSpec("Final Projection", "dense", "3D -> D"),
                LayerSpec("Grouped Score Head", "output", "3 scores"),
            ),
        ),
        ModelVizSpec(
            key="transformer_bertcounter_lec",
            title="Latent Edit Competition",
            family="counterfactual",
            summary="Shared latent edit vector competes across candidate repairs and explanations.",
            layers=(
                LayerSpec("Grouped Token IDs", "input", "[batch, 3, seq]"),
                LayerSpec("Shared Encoder", "attention", "DeBERTa-like x4"),
                LayerSpec("Latent Edit Module", "competition", "shared anomaly + edit vector"),
                LayerSpec("Option Competition", "competition", "explanation-first ranking"),
                LayerSpec("Final Projection", "dense", "4D -> D"),
                LayerSpec("Grouped Score Head", "output", "3 scores"),
            ),
        ),
    )


def get_model_specs() -> tuple[ModelVizSpec, ...]:
    return _specs()


def get_model_spec_map() -> dict[str, ModelVizSpec]:
    return {spec.key: spec for spec in get_model_specs()}


def selected_specs(model_names: list[str] | None) -> tuple[ModelVizSpec, ...]:
    specs = get_model_specs()
    if not model_names:
        return specs
    wanted = set(model_names)
    return tuple(spec for spec in specs if spec.key in wanted)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
