#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

run_python() {
  local label="$1"
  local module_name="$2"
  echo
  echo "=== ${label} ==="
  python -m "$module_name"
}

run_python "Transformer BBPE" "src.train_model_transformer"
run_python "Transformer AttentionTypes" "src.train_model_transformer_attentiontypes"
run_python "BertCounterFact" "src.train_model_bertcounterfact"
run_python "BertCounterFact Cross Option" "src.train_model_bertcounterfact_cross_option"
run_python "BertCounterFact Latent Edit Competition" "src.train_model_bertcounterfact_latent_edit_competition"

echo
echo "Transformer-family runs completed."
