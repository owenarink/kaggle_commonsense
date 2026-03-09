# trains all models
#!/usr/bin/env bash
set -euo pipefail

python -m src.train_model.py
python -m src.train_model_wPairwise 
python -m src.models.cnn_text
python -m src.train_model_cnn.py

# build with chmod +x run/run_all.sh

