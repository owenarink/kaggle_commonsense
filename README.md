# Commonsense Reasoning with Neural Networks

Course project for a Kaggle-style commonsense reasoning challenge. The task is to select the correct answer (`A`, `B`, or `C`) for a false sentence by comparing candidate options and training neural models to recover the sensible statement.

This repository contains baseline models, transformer-based experiments, custom tokenization, evaluation scripts, saved checkpoints, and analysis/visualization outputs. The main model explored here is a grouped BBPE transformer with DeBERTa-inspired disentangled attention.

## Highlights

- Multiple baselines and transformer variants for commonsense reasoning
- Grouped BBPE input formulation for multiple-choice scoring
- DeBERTa-inspired attention-type experiments
- Training diagnostics, architecture plots, and analysis figures included

## Result

The strongest final experiment in this repo is a grouped BBPE transformer with DeBERTa-like attention-type mechanisms, which achieved a Kaggle public score of about `0.7456`.

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

## Selected Figures

**AttentionTypes model architecture**

![AttentionTypes model](outputs/plots/architectures/drawio/transformer_attentiontypes_drawio_preview.png)

**Loss landscape**

![Loss landscape](outputs/plots/loss_landscapes/transformer_attentiontypes_loss_landscape.png)

**Parameter breakdown**

![Parameter breakdown](outputs/plots/readme/attentiontypes_parameter_breakdown.png)

**AdamW learning-rate schedule**

![Learning-rate schedule](outputs/plots/readme/attentiontypes_lr_schedule.png)

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
