from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from src.analysis.explainability_utils import LABELS, build_mlp_tfidf_bundle, load_mlp_model, predict_mlp_proba


def _shap_module():
    try:
        import shap
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install `shap` to run this script: pip install shap") from exc
    return shap


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for the trained TF-IDF MLP.")
    parser.add_argument("--sample-count", type=int, default=32, help="Number of validation samples to explain.")
    parser.add_argument("--background-count", type=int, default=32, help="Number of background samples.")
    parser.add_argument("--feature-count", type=int, default=128, help="Number of TF-IDF features kept for SHAP.")
    parser.add_argument("--output-dir", default="outputs/plots/explanations/shap")
    return parser.parse_args()


def _select_feature_subset(x_train: np.ndarray, x_val: np.ndarray, feature_count: int):
    scores = np.abs(x_train).mean(axis=0) + np.abs(x_val).mean(axis=0)
    top_idx = np.argsort(scores)[-feature_count:]
    top_idx = np.sort(top_idx)
    return top_idx


def _class_shap_values(shap_values, class_index: int):
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index])
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        if shap_values.shape[0] == class_index + 1 and shap_values.shape[1] != shap_values.shape[2]:
            return shap_values[class_index]
        return shap_values[:, :, class_index]
    return shap_values


def _make_fullspace_predictor(model, feature_idx: np.ndarray, full_dim: int):
    def predict(arr: np.ndarray) -> np.ndarray:
        full = np.zeros((arr.shape[0], full_dim), dtype=np.float32)
        full[:, feature_idx] = arr.astype(np.float32)
        return predict_mlp_proba(model, full)

    return predict


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_mlp_tfidf_bundle()
    model = load_mlp_model(bundle["x_train"].shape[1])
    shap = _shap_module()

    background_full = bundle["x_train"][: args.background_count]
    samples_full = bundle["x_val"][: args.sample_count]
    feature_idx = _select_feature_subset(bundle["x_train"], bundle["x_val"], args.feature_count)
    background = background_full[:, feature_idx]
    samples = samples_full[:, feature_idx]
    feature_names = bundle["feature_names"][feature_idx]
    predictor = _make_fullspace_predictor(model, feature_idx, bundle["x_train"].shape[1])
    probs = predictor(samples)
    pred_idx = int(np.argmax(probs.mean(axis=0)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        explainer = shap.KernelExplainer(predictor, background)
        shap_values = explainer.shap_values(
            samples,
            nsamples=min(2 * samples.shape[1] + 8, 256),
            l1_reg="num_features(15)",
        )
    class_values = _class_shap_values(shap_values, pred_idx)

    summary_path = output_dir / f"shap_summary_class_{LABELS[pred_idx]}.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        class_values,
        samples,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(summary_path, dpi=220, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(class_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-15:][::-1]
    top_names = feature_names[top_idx]
    top_vals = mean_abs[top_idx]
    bar_path = output_dir / f"shap_top_features_class_{LABELS[pred_idx]}.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(top_vals)), top_vals[::-1], color="#457B9D")
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_title(f"Mean |SHAP| top features for class {LABELS[pred_idx]}")
    ax.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout()
    fig.savefig(bar_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] summary -> {summary_path}")
    print(f"[ok] bar     -> {bar_path}")


if __name__ == "__main__":
    main()
