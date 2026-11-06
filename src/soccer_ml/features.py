"""Reusable feature selection and training-only preprocessing."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age", "height_cm", "weight_kg", "market_value_eur", "goals_per_match",
    "goals_per_season", "assists_per_match", "minutes_played", "matches_won",
    "annual_wage_eur",
]
CATEGORICAL_FEATURES = ["position", "league"]


def build_preprocessor(columns):
    """Impute and scale numbers; encode categories, including unseen leagues."""
    numeric = [name for name in NUMERIC_FEATURES if name in columns]
    categorical = [name for name in CATEGORICAL_FEATURES if name in columns]
    if not numeric and not categorical:
        raise ValueError("No supported feature columns were provided.")
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
