"""A small registry of classifiers with reproducible starting settings."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_models(seed=42, training_rows=5):
    """Return fresh estimators for each experiment, including a baseline."""
    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "adaboost": AdaBoostClassifier(random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=min(5, training_rows)),
        "svm": SVC(),
    }
