"""Build notebook tables from local Transfermarkt CSV exports (no network)."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LEAGUES = {"GB1": "Premier League", "FR1": "Ligue 1", "IT1": "Serie A",
           "L1": "Bundesliga", "ES1": "La Liga"}


def locate(root, name):
    matches = [p for p in root.rglob(name + ".csv") if p.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name}.csv below {root}; found {len(matches)}")
    return matches[0]


def read_selected(root, name, key=None, ids=None, chunksize=200000):
    path = locate(root, name)
    if key is None:
        return pd.read_csv(path, low_memory=False)
    parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        parts.append(chunk.loc[chunk[key].isin(ids)])
    return pd.concat(parts, ignore_index=True)


def unique(frame, keys, name):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{name}: missing or duplicate keys {keys}; resolve these before joining")
    return frame


def add_values(frame, valuations, date_column, prefix):
    """Use strictly earlier valuation dates; never fill backwards from the future."""
    values = valuations[["player_id", "date", "market_value_in_eur"]].copy()
    values = values.rename(columns={"date": prefix + "date", "market_value_in_eur": prefix + "eur"})
    values[prefix + "date"] = pd.to_datetime(values[prefix + "date"], errors="raise")
    values = unique(values, ["player_id", prefix + "date"], "valuations")
    left = frame.copy()
    left[date_column] = pd.to_datetime(left[date_column])
    result = pd.merge_asof(left.sort_values(date_column), values.sort_values(prefix + "date"),
        by="player_id", left_on=date_column, right_on=prefix + "date",
        direction="backward", allow_exact_matches=False)
    result[prefix + "age_days"] = (result[date_column] - result[prefix + "date"]).dt.days
    return result


def add_prior_features(frame):
    """Previous appearance totals within the same player/league/season."""
    frame = frame.sort_values(["player_id", "competition_id", "season_start_year", "date", "game_id"]).copy()
    groups = frame.groupby(["player_id", "competition_id", "season_start_year"], sort=False)
    frame["prior_appearances"] = groups.cumcount()
    for column in ["goals", "assists", "minutes_played"]:
        frame["prior_" + column] = groups[column].transform(lambda s: s.cumsum().shift(fill_value=0))
    frame["prior_goals_per90"] = 90 * frame["prior_goals"] / frame["prior_minutes_played"].replace(0, np.nan)
    frame["prior_assists_per90"] = 90 * frame["prior_assists"] / frame["prior_minutes_played"].replace(0, np.nan)
    return frame


def build_seasons(matches, valuations, by_club=False):
    keys = ["player_id", "competition_id", "league", "season_start_year", "season"]
    if by_club:
        keys += ["club_id", "club_name"]
    result = matches.groupby(keys, dropna=False).agg(
        player_name=("player_name", "first"), position=("position", "first"),
        sub_position=("sub_position", "first"), date_of_birth=("date_of_birth", "first"),
        height_cm=("height_cm", "first"), foot=("foot", "first"),
        first_match_date=("date", "min"), last_match_date=("date", "max"),
        appearances=("game_id", "nunique"), goals_per_season=("goals", lambda s: s.sum(min_count=1)),
        assists_per_season=("assists", lambda s: s.sum(min_count=1)),
        minutes_played=("minutes_played", lambda s: s.sum(min_count=1)),
        yellow_cards=("yellow_cards", lambda s: s.sum(min_count=1)),
        red_cards=("red_cards", lambda s: s.sum(min_count=1)),
        matches_won=("won", lambda s: s.sum(min_count=1)), clubs_represented=("club_id", "nunique"),
        missing_minutes_rows=("minutes_played", lambda s: s.isna().sum()),
        missing_goals_rows=("goals", lambda s: s.isna().sum()),
    ).reset_index()
    result["age"] = (result["first_match_date"] - result["date_of_birth"]).dt.days / 365.2425
    result["goals_per_match"] = result["goals_per_season"] / result["appearances"]
    result["assists_per_match"] = result["assists_per_season"] / result["appearances"]
    denominator = result["minutes_played"].replace(0, np.nan)
    result["goals_per90"] = 90 * result["goals_per_season"] / denominator
    result["assists_per90"] = 90 * result["assists_per_season"] / denominator
    result = add_values(result, valuations, "first_match_date", "market_value_before_first_match_")
    if not by_club:
        # Exact following season within the SAME league. No future observation stays null.
        future = result[["player_id", "competition_id", "season_start_year", "goals_per_season",
                         "assists_per_season", "minutes_played", "appearances"]].copy()
        future["season_start_year"] -= 1
        future = future.rename(columns={c: "target_next_season_" + c for c in
            ["goals_per_season", "assists_per_season", "minutes_played", "appearances"]})
        result = result.merge(future, on=["player_id", "competition_id", "season_start_year"],
                              how="left", validate="one_to_one")
    return result


def prepare(input_dir, output_dir, start_year=2010, end_year=None, competition_ids=None, chunksize=200000):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    competition_ids = list(LEAGUES) if competition_ids is None else competition_ids
    games = read_selected(input_dir, "games")
    games["date"] = pd.to_datetime(games["date"], errors="raise")
    games["season"] = pd.to_numeric(games["season"], errors="raise").astype(int)
    available = games.loc[games.competition_id.isin(competition_ids)].groupby("competition_id").season.agg(["min", "max"])
    games = games.loc[games.competition_id.isin(competition_ids) & games.season.ge(start_year)]
    if end_year is not None:
        games = games.loc[games.season.le(end_year)]
    games = unique(games.copy(), ["game_id"], "games")
    if games.empty:
        raise ValueError("No matches in the selected leagues/years")
    competitions = unique(read_selected(input_dir, "competitions"), ["competition_id"], "competitions")
    league_names = competitions.set_index("competition_id")["name"].to_dict() | LEAGUES
    games = games.rename(columns={"season": "season_start_year"})
    games["season"] = games.season_start_year.map(lambda y: f"{y}-{y+1}")
    games["league"] = games.competition_id.map(league_names)
    print(f"Selected {len(games):,} matches. Reading player appearances...", flush=True)
    appearances = read_selected(input_dir, "appearances", "game_id", games.game_id, chunksize)
    appearances = unique(appearances, ["game_id", "player_id"], "appearances")
    players = unique(read_selected(input_dir, "players"), ["player_id"], "players")
    valuations = read_selected(input_dir, "player_valuations", "player_id", appearances.player_id.unique(), chunksize)
    valuations["date"] = pd.to_datetime(valuations["date"], errors="raise")
    unique(valuations, ["player_id", "date"], "valuations")
    # Use match-time club membership, never the player's current club.
    base = appearances.drop(columns=["date", "competition_id", "player_current_club_id"])
    base = base.rename(columns={"player_club_id": "club_id"})
    match_columns = ["game_id", "competition_id", "league", "season_start_year", "season", "date",
                     "home_club_id", "away_club_id", "home_club_goals", "away_club_goals"]
    matches = base.merge(games[match_columns], on="game_id", how="left", validate="many_to_one")
    profile_columns = ["player_id", "date_of_birth", "position", "sub_position", "foot", "height_in_cm"]
    matches = matches.merge(players[profile_columns], on="player_id", how="left", validate="many_to_one")
    matches = matches.rename(columns={"height_in_cm": "height_cm"})
    matches["date_of_birth"] = pd.to_datetime(matches.date_of_birth, errors="raise")
    matches["age"] = (matches.date - matches.date_of_birth).dt.days / 365.2425
    home = matches.club_id.eq(matches.home_club_id)
    away = matches.club_id.eq(matches.away_club_id)
    if not (home | away).all():
        raise ValueError("Some appearance clubs match neither team in the game")
    matches["is_home"] = home.astype(int)
    matches["opponent_id"] = np.where(home, matches.away_club_id, matches.home_club_id)
    matches["team_goals"] = np.where(home, matches.home_club_goals, matches.away_club_goals)
    matches["opponent_goals"] = np.where(home, matches.away_club_goals, matches.home_club_goals)
    known = matches.team_goals.notna() & matches.opponent_goals.notna()
    matches["match_result"] = pd.Series(pd.NA, index=matches.index, dtype="string")
    matches.loc[known, "match_result"] = np.select(
        [matches.loc[known, "team_goals"].gt(matches.loc[known, "opponent_goals"]),
         matches.loc[known, "team_goals"].lt(matches.loc[known, "opponent_goals"])], ["W", "L"], default="D")
    matches["won"] = matches.match_result.eq("W").astype("Int64")
    matches["target_scored"] = matches.goals.gt(0).astype("Int64").mask(matches.goals.isna())
    clubs = unique(read_selected(input_dir, "clubs"), ["club_id"], "clubs")
    club_names = clubs.set_index("club_id")["name"]
    matches["club_name"] = matches.club_id.map(club_names)
    matches["opponent_name"] = matches.opponent_id.map(club_names)
    print("Reading match lineups and matching historical valuations...", flush=True)
    lineups = read_selected(input_dir, "game_lineups", "game_id", games.game_id, chunksize)
    unique(lineups, ["game_id", "player_id"], "lineups")
    lineup_columns = {"type": "lineup_type", "position": "match_position", "number": "shirt_number"}
    lineups = lineups.rename(columns=lineup_columns)
    matches = matches.merge(lineups[["game_id", "player_id", "lineup_type", "match_position", "shirt_number", "team_captain"]],
                            on=["game_id", "player_id"], how="left", validate="one_to_one")
    matches = add_values(matches, valuations, "date", "market_value_before_match_")
    matches = add_prior_features(matches)
    seasons = build_seasons(matches, valuations)
    club_seasons = build_seasons(matches, valuations, by_club=True)
    team_matches = read_selected(input_dir, "club_games", "game_id", games.game_id, chunksize)
    unique(team_matches, ["game_id", "club_id"], "club_games")
    team_matches = team_matches.merge(games[["game_id", "date", "season", "season_start_year", "competition_id", "league"]],
        on="game_id", validate="many_to_one")
    team_matches["club_name"] = team_matches.club_id.map(club_names)
    team_matches["opponent_name"] = team_matches.opponent_id.map(club_names)
    team_matches["match_result"] = np.select([team_matches.own_goals.gt(team_matches.opponent_goals),
        team_matches.own_goals.lt(team_matches.opponent_goals)], ["W", "L"], default="D")
    team_matches.loc[team_matches.own_goals.isna() | team_matches.opponent_goals.isna(), "match_result"] = pd.NA
    tables = {"player_match": matches, "player_season": seasons, "player_club_season": club_seasons,
              "team_match": team_matches, "matches": games, "players": players,
              "clubs": clubs, "competitions": competitions, "lineups": lineups,
              "player_valuations": valuations}
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary = []
    db_temp = output_dir / "transfermarkt.sqlite.tmp"
    print("Writing CSV tables and SQLite database...", flush=True)
    with closing(sqlite3.connect(db_temp)) as connection:
        for name, frame in tables.items():
            # CSV remains a portable alternative to SQLite; no optional parquet engine needed.
            temp = output_dir / (name + ".csv.tmp")
            frame.to_csv(temp, index=False)
            temp.replace(output_dir / (name + ".csv"))
            sql_frame = frame.copy()
            for c in sql_frame.select_dtypes(include="datetime").columns:
                sql_frame[c] = sql_frame[c].dt.strftime("%Y-%m-%d")
            sql_frame.to_sql(name, connection, index=False, if_exists="replace", chunksize=10000)
            for column in frame:
                dictionary.append({"table": name, "column": column, "dtype": str(frame[column].dtype),
                    "missing_rows": int(frame[column].isna().sum()),
                    "role": column_role(name, column)})
            print(f"  {name}: {len(frame):,} rows", flush=True)
        for table in ["player_match", "player_season", "player_club_season", "player_valuations"]:
            connection.execute(f'CREATE INDEX "{table}_player" ON "{table}" (player_id)')
        connection.execute('CREATE INDEX player_match_date ON player_match (date)')
        connection.commit()
    db_temp.replace(output_dir / "transfermarkt.sqlite")
    pd.DataFrame(dictionary).to_csv(output_dir / "data_dictionary.csv", index=False)
    coverage = matches.groupby(["competition_id", "league", "season_start_year"]).agg(
        games_with_appearances=("game_id", "nunique"), players=("player_id", "nunique"),
        appearances=("game_id", "size"), missing_historical_values=("market_value_before_match_eur", lambda s: s.isna().sum()))
    coverage.to_csv(output_dir / "coverage.csv")
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "input_dir": str(input_dir.resolve()),
        "requested_start_year": start_year, "requested_end_year": end_year,
        "competition_ids": competition_ids, "source_season_ranges": available.reset_index().to_dict("records"),
        "tables": {name: len(frame) for name, frame in tables.items()},
        "games_without_appearances": int((~games.game_id.isin(matches.game_id)).sum()),
        "valuation_rule": "Latest date strictly BEFORE match or first observed season appearance; no future filling.",
        "profile_caveat": "Position, height and foot are snapshot attributes, not historically versioned.",
        "missing_competitions": sorted(set(competition_ids) - set(available.index)),
        "sources": {name: {"path": str(locate(input_dir, name)), "bytes": locate(input_dir, name).stat().st_size}
                    for name in ["games", "appearances", "players", "player_valuations", "game_lineups", "club_games", "clubs", "competitions"]}}
    (output_dir / "build_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def column_role(table, column):
    if column.startswith("target_"):
        return "target: future or current outcome; never include in predictors"
    if column.startswith("prior_") or column.startswith("market_value_before_"):
        return "historical feature; verify observation dates for your task"
    if column in {"position", "sub_position", "height_cm", "foot"}:
        return "snapshot profile; not historically versioned"
    if table in {"player_match", "player_season", "player_club_season", "team_match", "matches"}:
        return "identifier/context or contemporaneous statistic; review before prediction"
    return "source reference data; review before prediction"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/raw/transfermarkt-datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/transfermarkt")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--competitions", nargs="+", default=list(LEAGUES))
    parser.add_argument("--chunksize", type=int, default=200000)
    args = parser.parse_args(argv)
    if args.chunksize < 1 or (args.end_year is not None and args.end_year < args.start_year):
        parser.error("Use a positive chunksize and end-year >= start-year")
    metadata = prepare(args.input_dir, args.output_dir, args.start_year, args.end_year, args.competitions, args.chunksize)
    print(f"Ready for notebooks: {args.output_dir.resolve()}")
    print(f"Games without appearance records: {metadata['games_without_appearances']:,}")
    for row in metadata["source_season_ranges"]:
        print(f"Source coverage {row['competition_id']}: {row['min']} through {row['max']}")


if __name__ == "__main__":
    main()
