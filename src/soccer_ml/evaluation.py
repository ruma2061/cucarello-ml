"""Classification metrics using a shared label order."""

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def evaluate(y_true, y_pred, labels):
    """Macro averages give each class equal weight; recall is sensitivity."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "labels": list(labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
