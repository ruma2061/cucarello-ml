"""Checks for historical joins, past-only features, and transfer aggregation."""
import unittest
import tempfile
from pathlib import Path
import sqlite3

import pandas as pd

from soccer_ml.prepare_transfermarkt import ROOT, add_prior_features, add_values, build_seasons, prepare, unique


class TransfermarktTests(unittest.TestCase):
    def test_values_never_use_same_day_or_future(self):
        rows = pd.DataFrame({"player_id": [1, 1, 2], "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-01"])})
        values = pd.DataFrame({"player_id": [1, 1, 2], "date": ["2020-01-01", "2020-01-02", "2021-01-01"],
                               "market_value_in_eur": [100, 200, 999]})
        result = add_values(rows, values, "date", "value_").set_index(["player_id", "date"])
        self.assertEqual(result.loc[(1, pd.Timestamp("2020-01-02")), "value_eur"], 100)
        self.assertEqual(result.loc[(1, pd.Timestamp("2020-01-03")), "value_eur"], 200)
        self.assertTrue(pd.isna(result.loc[(2, pd.Timestamp("2020-01-01")), "value_eur"]))

    def test_prior_totals_exclude_current_match_and_reset_season(self):
        frame = pd.DataFrame({"player_id": [1, 1, 1], "competition_id": ["GB1"] * 3,
            "season_start_year": [2020, 2020, 2021], "date": pd.to_datetime(["2020-08-01", "2020-08-08", "2021-08-01"]),
            "game_id": [1, 2, 3], "goals": [2, 5, 1], "assists": [1, 2, 0], "minutes_played": [90, 80, 90]})
        result = add_prior_features(frame)
        self.assertEqual(result.prior_goals.tolist(), [0, 2, 0])
        self.assertEqual(result.prior_minutes_played.tolist(), [0, 90, 0])

    def test_transfers_combine_and_target_requires_exact_next_season(self):
        frame = pd.DataFrame({"player_id": [1, 1, 1, 1], "competition_id": ["GB1"] * 4,
            "league": ["Premier League"] * 4, "season_start_year": [2020, 2020, 2021, 2023],
            "season": ["2020-2021", "2020-2021", "2021-2022", "2023-2024"],
            "club_id": [10, 20, 20, 20], "club_name": ["A", "B", "B", "B"],
            "player_name": ["Player"] * 4, "position": ["Attack"] * 4, "sub_position": ["Forward"] * 4,
            "date_of_birth": pd.to_datetime(["2000-01-01"] * 4), "height_cm": [180] * 4, "foot": ["right"] * 4,
            "date": pd.to_datetime(["2020-08-01", "2021-02-01", "2021-08-01", "2023-08-01"]),
            "game_id": [1, 2, 3, 4], "goals": [2, 3, 7, 9], "assists": [0] * 4,
            "minutes_played": [90] * 4, "yellow_cards": [0] * 4, "red_cards": [0] * 4, "won": [1] * 4})
        values = pd.DataFrame({"player_id": [1], "date": ["2020-01-01"], "market_value_in_eur": [100]})
        seasons = build_seasons(frame, values).set_index("season_start_year")
        self.assertEqual(seasons.loc[2020, "goals_per_season"], 5)
        self.assertEqual(seasons.loc[2020, "clubs_represented"], 2)
        self.assertEqual(seasons.loc[2020, "target_next_season_goals_per_season"], 7)
        self.assertTrue(pd.isna(seasons.loc[2021, "target_next_season_goals_per_season"]))
        self.assertEqual(len(build_seasons(frame, values, by_club=True)), 4)

    def test_duplicate_keys_fail_before_expanding_joins(self):
        with self.assertRaises(ValueError):
            unique(pd.DataFrame({"id": [1, 1]}), ["id"], "example")

    def test_complete_build_and_rebuild_close_database(self):
        tables = {
            "games": pd.DataFrame({"game_id": [1], "competition_id": ["GB1"], "season": [2020],
                "date": ["2020-08-01"], "home_club_id": [10], "away_club_id": [20],
                "home_club_goals": [1], "away_club_goals": [0]}),
            "appearances": pd.DataFrame({"game_id": [1], "player_id": [1], "player_club_id": [10],
                "player_current_club_id": [20], "date": ["2020-08-01"], "competition_id": ["GB1"],
                "player_name": ["P"], "goals": [1], "assists": [0], "minutes_played": [90],
                "yellow_cards": [0], "red_cards": [0]}),
            "players": pd.DataFrame({"player_id": [1], "date_of_birth": ["2000-01-01"],
                "position": ["Attack"], "sub_position": ["Forward"], "foot": ["right"], "height_in_cm": [180]}),
            "player_valuations": pd.DataFrame({"player_id": [1], "date": ["2020-07-01"], "market_value_in_eur": [100]}),
            "game_lineups": pd.DataFrame({"game_id": [1], "player_id": [1], "club_id": [10],
                "type": ["starting_lineup"], "position": ["Forward"], "number": [9], "team_captain": [0]}),
            "clubs": pd.DataFrame({"club_id": [10, 20], "name": ["A", "B"]}),
            "club_games": pd.DataFrame({"game_id": [1, 1], "club_id": [10, 20],
                "opponent_id": [20, 10], "own_goals": [1, 0], "opponent_goals": [0, 1]}),
            "competitions": pd.DataFrame({"competition_id": ["GB1"], "name": ["premier-league"]}),
        }
        with tempfile.TemporaryDirectory(dir=ROOT / "data/processed") as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            for name, frame in tables.items():
                nested = raw / (name + ".csv")
                nested.mkdir()
                frame.to_csv(nested / (name + ".csv"), index=False)
            out = Path(tmp) / "tables"
            for _ in range(2):
                metadata = prepare(raw, out, competition_ids=["GB1"])
                self.assertEqual(metadata["tables"]["player_match"], 1)
                self.assertEqual(metadata["games_without_appearances"], 0)
            connection = sqlite3.connect(out / "transfermarkt.sqlite")
            try:
                match = pd.read_sql_query("SELECT * FROM player_match", connection).iloc[0]
                self.assertEqual(match.club_id, 10)  # historical club, not current club 20
                self.assertEqual(match.match_result, "W")
                self.assertEqual(match.market_value_before_match_eur, 100)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
