#  chmod +x run/run_cnn.sh
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


run_step_quiet  "1/2 src.models.cnn_text"                     python -m src.models.cnn_text
run_step_chatty "2/2 src.train_model_cnn"                     python -m src.train_model_cnn

echo "All steps completed. 🍺"
