from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.explainability_utils import LABELS, build_mlp_tfidf_bundle, load_mlp_model, predict_mlp_proba


def _lime_module():
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install `lime` to run this script: pip install lime") from exc
    return LimeTabularExplainer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate LIME explanations for the trained TF-IDF MLP.")
    parser.add_argument("--sample-index", type=int, default=0, help="Validation sample index to explain.")
    parser.add_argument("--num-features", type=int, default=12, help="Number of features to show.")
    parser.add_argument("--output-dir", default="outputs/plots/explanations/lime")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_mlp_tfidf_bundle()
    model = load_mlp_model(bundle["x_train"].shape[1])
    explainer_cls = _lime_module()
    explainer = explainer_cls(
        bundle["x_train"],
        feature_names=bundle["feature_names"],
        class_names=LABELS,
        mode="classification",
        discretize_continuous=False,
    )

    index = max(0, min(args.sample_index, len(bundle["x_val"]) - 1))
    x = bundle["x_val"][index]
    probs = predict_mlp_proba(model, x[None, :])[0]
    pred_idx = int(np.argmax(probs))

    explanation = explainer.explain_instance(
        x,
        lambda arr: predict_mlp_proba(model, arr),
        num_features=args.num_features,
        top_labels=1,
    )

    html_path = output_dir / f"lime_mlp_sample_{index}.html"
    explanation.save_to_file(str(html_path))

    features = explanation.as_list(label=pred_idx)
    labels = [item[0] for item in features]
    values = [item[1] for item in features]
    colors = ["#2A9D8F" if value >= 0 else "#E76F51" for value in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"LIME explanation for validation sample {index}\nPredicted class: {LABELS[pred_idx]}")
    ax.set_xlabel("Contribution")
    fig.tight_layout()
    png_path = output_dir / f"lime_mlp_sample_{index}.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] html -> {html_path}")
    print(f"[ok] png  -> {png_path}")


if __name__ == "__main__":
    main()
