# Commonsense Reasoning with Neural Networks

Course project for a Kaggle-style commonsense reasoning challenge. The task is to select the correct answer (`A`, `B`, or `C`) for a false sentence by comparing candidate options and training neural models to recover the sensible statement.

This repository contains baseline models, transformer-based experiments, custom tokenization, evaluation scripts, saved checkpoints, and analysis/visualization outputs used for the project and poster.

## Visual Overview

**Model overview**

![Model overview](outputs/plots/poster_panels/transformer_encoder_stack_panel.png)

**Loss landscape**

![Loss landscape](outputs/plots/loss_landscapes/transformer_attentiontypes_loss_landscape.png)

## Project Focus

- Build and compare neural approaches for sentence-level commonsense reasoning
- Experiment with grouped input modeling for multiple-choice prediction
- Explore DeBERTa-inspired attention-type variants and BBPE tokenization
- Analyze model behavior with plots, architecture diagrams, and explainability scripts

## Repository Structure

- `src/`: training, evaluation, preprocessing, models, and analysis code
- `data/`: training/test data and sample submission files
- `checkpoints/`: saved model weights and tokenizer metadata
- `outputs/`: generated plots and architecture visualizations
- `docs/`: supporting notes

## Main Result

The strongest final experiment in this repo is a grouped BBPE transformer with DeBERTa-like attention-type mechanisms, which achieved a Kaggle public score of about `0.7456`.

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

## Notes

- Some scripts were written for local course experimentation and may assume existing checkpoints or prepared tokenizers
- The repository includes generated outputs and trained artifacts for reproducibility and project presentation

## Authors

- Owen Arink
- T. Santos Andersen
