
#!/usr/bin/env bash
set -euo pipefail
python -m src.analysis.plot_confusion_mlp_pairwise 
python -m src.analysis.plot_feature_stats_mlp_pairwise 
python -m src.analysis.plot_confusion 
python -m src.analysis.plot_feature_stats 
python -m src.analysis.plot_loss_landscapes
python -m src.analysis.print_category_accuracy
python -m src.analysis.print_pairwise_category_accuracy
