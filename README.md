# Commonsense Reasoning with Neural Networks

Course project for a Kaggle-style commonsense reasoning challenge. The task is to select the correct answer (`A`, `B`, or `C`) for a false sentence by comparing candidate options and training neural models to recover the sensible statement.

This repository contains baseline models, transformer-based experiments, custom tokenization, evaluation scripts, saved checkpoints, and analysis/visualization outputs. The main model explored here is a grouped BBPE transformer with DeBERTa-inspired disentangled attention.

## Highlights

- Multiple baselines and transformer variants for commonsense reasoning
- Grouped BBPE input formulation for multiple-choice scoring
- DeBERTa-inspired attention-type experiments
- Training diagnostics, architecture plots, and analysis figures included

## Result

The strongest final experiment in this repo is the grouped BBPE `AttentionTypes` model. It reached a best validation accuracy of `0.7456` and a final Kaggle public score of `0.7350`.

## Competition-Style Summary

- Input: one false sentence with three candidate corrections
- Representation: grouped BBPE tokenization
- Model: shared transformer encoder with disentangled attention
- Pooling: mean pooling over valid tokens
- Optimizer: AdamW with warmup + cosine decay
- Output: one score per option, then `argmax` over `A/B/C`

## Core Equations

For each question, the model scores the three candidate options and predicts the highest-scoring answer:

$$
\hat{y} = \arg\max_{k \in \{A,B,C\}} s_k
$$

Each option is passed through the same encoder and linear score head:

$$
h_k = \mathrm{Encoder}(x_k), \qquad s_k = W h_k + b
$$

The grouped training objective is cross-entropy over the three option scores:

$$
\mathcal{L}_{\mathrm{CE}} = - \log \frac{\exp(s_y)}{\sum_{k=1}^{3} \exp(s_k)}
$$

The simplified DeBERTa-style attention in this project mixes three score terms:

$$
\mathrm{score}_{ij} = q_i^\top k_j + q_i^\top r_{ij}^{(K)} + r_{ij}^{(Q)\top} k_j
$$

and the final attention weights are

$$
\alpha_{ij} = \mathrm{softmax}\left(\frac{\mathrm{score}_{ij}}{\sqrt{d_k \cdot 3}}\right)
$$

Relative position embeddings encode token distance rather than absolute position:

$$
r_{ij} = \mathrm{Embed}\big(\mathrm{clip}(j-i,\,-M,\,M)\big)
$$

Mean pooling averages only over non-padding tokens:

$$
h = \frac{\sum_{t=1}^{L} m_t x_t}{\sum_{t=1}^{L} m_t}
$$

where `m_t = 1` for real tokens and `0` for padding.

Dropout is used as regularization during training:

$$
\tilde{h} = \frac{m \odot h}{1-p}, \qquad m \sim \mathrm{Bernoulli}(1-p)
$$

The optimizer is AdamW, which combines adaptive moments with decoupled weight decay:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t
$$

## Other DeBERTa-Style Variants Tried

I also tried a few lighter and heavier relative-position attention variants around the main `AttentionTypes` model.

**1. RelativeSimple**

This removes the position-to-content term and keeps only content-to-content plus content-to-position:

$$
\mathrm{score}_{ij}^{\mathrm{relsimple}} = q_i^\top k_j + q_i^\top r_{ij}^{(K)}
$$

This is simpler than full disentangled attention and tests whether relative keys alone are enough.

**2. SepDistance**

This adds an extra term based on distance from the separator token between the false sentence and candidate answer:

$$
\mathrm{score}_{ij}^{\mathrm{sepdist}} =
q_i^\top k_j + q_i^\top r_{ij}^{(K)} + r_{ij}^{(Q)\top} k_j + q_i^\top s_j
$$

where `s_j` is an embedding of the token's distance to the `[SEP]` boundary. The idea is to tell the model more explicitly whether a token belongs near the false sentence side or the candidate-answer side.

**3. AttentionTypes**

This is the full variant used for the main result:

$$
\mathrm{score}_{ij}^{\mathrm{attn-types}} = q_i^\top k_j + q_i^\top r_{ij}^{(K)} + r_{ij}^{(Q)\top} k_j
$$

This turned out to be the strongest and most stable of the DeBERTa-inspired attention variants in the repository.

## Selected Figures

**AttentionTypes model architecture**

![AttentionTypes model](outputs/plots/architectures/drawio/transformer_attentiontypes_drawio_preview.png)

**3D loss diagnostics**

<table>
  <tr>
    <td><img src="outputs/plots/loss_landscapes/transformer_attentiontypes_loss_landscape.png" alt="Loss landscape" width="100%"></td>
    <td><img src="outputs/plots/readme/attentiontypes_loss_basin.png" alt="Loss basin" width="100%"></td>
    <td><img src="outputs/plots/readme/attentiontypes_sharpness_surface.png" alt="Sharpness surface" width="100%"></td>
  </tr>
  <tr>
    <td align="center">Original loss landscape</td>
    <td align="center">Basin view around the minimum</td>
    <td align="center">Gradient-magnitude surface</td>
  </tr>
</table>

**Optimization diagnostics**

<table>
  <tr>
    <td><img src="outputs/plots/readme/attentiontypes_parameter_breakdown.png" alt="Parameter breakdown" width="100%"></td>
    <td><img src="outputs/plots/readme/attentiontypes_lr_schedule.png" alt="Learning-rate schedule" width="100%"></td>
  </tr>
  <tr>
    <td align="center">Where model capacity is allocated</td>
    <td align="center">AdamW warmup-cosine schedule</td>
  </tr>
</table>

## How The Model Works

- The model reads each `(false sentence, option)` pair separately, but uses the same encoder for all three options.
- DeBERTa-style attention helps the model keep apart two things: what the token means and where it is relative to another token.
- Relative position embeddings tell the model whether another token is nearby, far away, before, or after the current token.
- Mean pooling turns the full token sequence into one vector by averaging the valid token representations.
- The final linear layer converts that pooled vector into one score per answer option.

## Hyperparameters

The main `attentiontypes` experiment uses:

- `model_dim = 256`
- `num_heads = 8`
- `num_layers = 4`
- `ff_mult = 4`
- `dropout = 0.30`
- `grouped_max_len = 128`
- `optimizer = AdamW`
- `learning_rate = 6e-5`
- `weight_decay = 5e-2`
- `label_smoothing = 0.10`
- `batch_size = 32`
- `warmup = 5% of scheduled steps`
- `scheduler = cosine decay`

## Validation Results

The table below summarizes the archived validation scores from the saved comparison figure and training logs in this repository.

| Model | Validation accuracy |
| --- | ---: |
| MLP + TF-IDF | 0.3338 |
| Pairwise MLP | 0.7100 |
| Transformer BBPE | 0.7300 |
| AttentionTypes | 0.7456 |
| Counterfactual Repair | 0.7400 |
| Cross-Option Attention | 0.7344 |
| Latent Edit Competition | 0.7125 |

For the final competition submission, the `AttentionTypes` model achieved a Kaggle public score of `0.7350`.

## Repository Structure

- `src/`: training, evaluation, preprocessing, models, and analysis code
- `data/`: training/test data and sample submission files
- `checkpoints/`: saved model weights and tokenizer metadata
- `outputs/`: generated plots and architecture visualizations
- `docs/`: supporting notes

## Getting Started

Clone the repository and install the Python dependencies used in the project environment. Then run one of the training scripts from `src/`, for example:

```bash
python -m src.train_model_transformer_attentiontypes
```

Other entry points include:

```bash
python -m src.run_baselines
python -m src.train_model_transformer
python -m src.eval
```

To regenerate the README figures added in this page:

```bash
python -m src.analysis.generate_readme_figures
```

## Notes

- Some scripts were written for local course experimentation and may assume existing checkpoints or prepared tokenizers
- The repository includes generated outputs and trained artifacts for reproducibility and project presentation

## Authors

- Owen Arink
- T. Santos Andersen
