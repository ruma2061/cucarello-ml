"""Load prepared player-season tables and filter comparison groups."""

import pandas as pd


def load_dataset(path, target):
    """Read a CSV whose outcome labels have already been defined."""
    frame = pd.read_csv(path)
    required = {"player_id", "season", "league", "position", target}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Dataset must contain rows and complete identifiers/outcome labels.")
    return frame


def filter_players(frame, position=None, league=None):
    """Optionally restrict an experiment to a position and/or league."""
    if position is not None:
        frame = frame.loc[frame["position"] == position]
    if league is not None:
        frame = frame.loc[frame["league"] == league]
    if frame.empty:
        raise ValueError("No rows match the selected filters.")
    return frame.copy()
