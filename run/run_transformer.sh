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

run_step_quiet  "1/5 src.processing"                          python -m src.processing
run_step_quiet  "2/5 src.features.tfidf"                      python -m src.features.tfidf

# chatty training steps (won't glue ball onto epoch lines)
run_step_chatty "3/5 src.models.transformer" python -m src.models.transformer
run_step_chatty "4/5 src.train_model_transformer"                         python -m src.train_model_transformer


run_step_quiet  "5/5 submits"                     python -m src.submits
echo "All steps completed. 🍺"
