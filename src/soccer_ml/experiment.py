"""Run baseline comparisons against a held-out season or league."""

import argparse
import json
from pathlib import Path

from sklearn.pipeline import Pipeline

from soccer_ml.data import filter_players, load_dataset
from soccer_ml.evaluation import evaluate
from soccer_ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor
from soccer_ml.models import get_models


def run_experiment(frame, target, test_season, held_out_league=None):
    """Train on earlier seasons; optionally test transfer to an unseen league."""
    seasons = frame["season"].astype(str)
    train = frame.loc[seasons < test_season]
    test = frame.loc[seasons == test_season]
    if held_out_league:
        train = train.loc[train["league"] != held_out_league]
        test = test.loc[test["league"] == held_out_league]
    if train.empty or test.empty:
        raise ValueError("Need earlier training seasons and rows in the selected test season/league.")
    if train[target].nunique() < 2:
        raise ValueError("Training data must contain at least two outcome classes.")
    features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in frame and c != target]
    labels = sorted(frame[target].unique().tolist())
    results = {}
    for name, estimator in get_models(training_rows=len(train)).items():
        pipeline = Pipeline([
            ("prepare", build_preprocessor(features)),
            ("model", estimator),
        ])
        pipeline.fit(train[features], train[target])
        results[name] = evaluate(test[target], pipeline.predict(test[features]), labels)
    return {
        "target": target, "test_season": test_season,
        "held_out_league": held_out_league, "features": features,
        "training_rows": len(train), "test_rows": len(test), "models": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--test-season", required=True)
    parser.add_argument("--position")
    parser.add_argument("--league")
    parser.add_argument("--held-out-league")
    parser.add_argument("--output", type=Path, default=Path("reports/metrics.json"))
    args = parser.parse_args()
    frame = filter_players(load_dataset(args.data, args.target), args.position, args.league)
    results = run_experiment(frame, args.target, args.test_season, args.held_out_league)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Compared {len(results['models'])} models. Results: {args.output}")


if __name__ == "__main__":
    main()
