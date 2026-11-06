# Experiment reports

Each run saves accuracy, macro recall, macro F1, and confusion matrices to JSON.
Confusion matrix rows are actual labels; columns are predicted labels, in `labels` order.
Use a different `--output` path to keep each experiment's results.
Model settings currently live in `src/soccer_ml/models.py`.
