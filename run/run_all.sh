#  chmod +x run/run_all.sh
#!/usr/bin/env bash
set -euo pipefail

spinner() {
  local pid="$1"
  local msg="$2"
  local frames=( "a" "b" "c" "d" )
  local i=0

  printf "%s " "$msg"
  while kill -0 "$pid" 2>/dev/null; do
    i=$(( (i + 1) % ${#frames[@]} ))
    printf "\r%s %s" "$msg" "${frames[$i]}"
    sleep 0.12
  done
  printf "\r%s 🍺\n\n" "$msg"
}

run_step_quiet() {
  local msg="$1"
  shift
  ("$@") & local pid=$!
  spinner "$pid" "$msg"
  wait "$pid"
}

# Use this for commands that print lots of output (epochs, tqdm, etc.)
run_step_chatty() {
  local msg="$1"
  shift
  echo "${msg} started"
  echo
  "$@"
  echo
  echo "${msg} 🍺"
  echo
}

run_step_quiet  "1/12 src.processing"                          python -m src.processing
run_step_quiet  "2/12 src.features.tfidf"                      python -m src.features.tfidf

# chatty training steps (won't glue ball onto epoch lines)
run_step_chatty "3/12 src.train_model"                         python -m src.train_model
run_step_chatty "4/12 src.train_model_wPairwise"               python -m src.train_model_wPairwise

run_step_quiet  "5/12 src.models.cnn_text"                     python -m src.models.cnn_text
run_step_chatty "6/12 src.train_model_cnn"                     python -m src.train_model_cnn

run_step_quiet  "7/12 plot_confusion_mlp_pairwise"             python -m src.analysis.plot_confusion_mlp_pairwise
run_step_quiet  "8/12 plot_feature_stats_mlp_pairwise"         python -m src.analysis.plot_feature_stats_mlp_pairwise
run_step_quiet  "9/12 plot_confusion"                          python -m src.analysis.plot_confusion
run_step_quiet  "10/12 plot_feature_stats"                     python -m src.analysis.plot_feature_stats
run_step_quiet  "11/12 print_category_accuracy"                python -m src.analysis.print_category_accuracy
run_step_quiet  "12/12 print_pairwise_category_accuracy"       python -m src.analysis.print_pairwise_category_accuracy

echo "All steps completed. 🍺"
