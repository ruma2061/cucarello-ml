"""Offline checks for discovery, completeness, checkpoints, and CSV preservation."""

import json
from datetime import timedelta
from io import BytesIO
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from soccer_ml.download_fbref import (
    AccessBlocked, ROOT, Store, both_teams, collect_season, export_frame,
    access_pause, make_reader, output_lock, parse_args, run_once, season_start, utc_now,
)


class FakeReader:
    calls = []
    leagues = ["ENG-Premier League"]
    seasons = ["1011"]

    def read_leagues(self):
        return pd.DataFrame({"url": ["/league"]}, index=pd.Index(self.leagues, name="league"))

    def discover_seasons(self):
        return pd.DataFrame({"url": ["/2010", "/2026"]}, index=pd.MultiIndex.from_tuples(
            [(self.leagues[0], "1011"), (self.leagues[0], "2627")], names=["league", "season"]))

    def read_schedule(self):
        return pd.DataFrame({"game_id": ["abc12345", None], "match_report": ["/en/matches/abc12345", None]},
            index=pd.MultiIndex.from_tuples([(self.leagues[0], "1011", "played"),
                (self.leagues[0], "1011", "upcoming")], names=["league", "season", "game"]))

    def _is_complete(self, league, season):
        return True

    def read_player_season_stats(self, stat):
        return pd.DataFrame({"goals": [1]})

    def read_team_season_stats(self, stat, opponent_stats=False):
        return pd.DataFrame({"goals": [1]})

    def read_team_match_stats(self, stat, opponent_stats=False, force_cache=False):
        return pd.DataFrame({"goals": [1]})

    def read_player_match_stats(self, stat, match_id=None, force_cache=False):
        self.calls.append((stat, match_id))
        return pd.DataFrame({"goals": [1, 0]}, index=pd.Index(["home", "away"], name="team"))

    def read_lineup(self, match_id=None, force_cache=False):
        return self.read_player_match_stats("lineup", match_id)

    def read_events(self, match_id=None, force_cache=False):
        return pd.DataFrame({"minute": [15], "event": ["goal"]})


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "data/raw")
        self.addCleanup(self.temp.cleanup)
        _, self.config = parse_args(["--output-dir", self.temp.name, "--start-year", "2010", "--end-year", "2010"])
        self.config["retries"] = 0
        FakeReader.calls = []

    def test_season_codes(self):
        for code, expected in [("1011", 2010), ("2021", 2020), ("2627", 2026),
                               ("2010-2011", 2010), ("2024", 2024), ("9900", 1999)]:
            self.assertEqual(season_start(code), expected)

    def test_resume_and_missing_export(self):
        store = Store(Path(self.temp.name), self.config)
        frame = pd.DataFrame({"x": [1]})
        store.fetch("test", lambda: frame)
        self.assertFalse(store.due("test"))
        (store.root / "test.csv").unlink()
        self.assertTrue(store.due("test"))
        store.fetch("empty", lambda: pd.DataFrame())
        self.assertEqual(store.state["empty"]["status"], "empty")
        self.assertFalse(store.due("empty"))

    def test_partial_and_errors_are_not_complete(self):
        store = Store(Path(self.temp.name), self.config)
        store.fetch("partial", lambda: pd.DataFrame({"x": [1]}), check=both_teams)
        self.assertEqual(store.state["partial"]["status"], "partial")
        self.assertFalse(store.due("partial"))
        def fail():
            raise ValueError("missing table")
        store.fetch("failure", fail)
        self.assertEqual(store.state["failure"]["status"], "error")
        self.assertFalse(store.due("failure"))
        store.state["failure"]["updated_at"] = (utc_now() - timedelta(days=2)).isoformat()
        self.assertTrue(store.due("failure"))
        with self.assertRaises(AccessBlocked):
            store.fetch("blocked", lambda: (_ for _ in ()).throw(AccessBlocked("403")))

    def test_refresh_and_empty_retry_intervals(self):
        store = Store(Path(self.temp.name), self.config)
        store.fetch("mutable", lambda: pd.DataFrame({"x": [1]}))
        store.state["mutable"]["updated_at"] = (utc_now() - timedelta(days=2)).isoformat()
        self.assertFalse(store.due("mutable"))
        self.assertTrue(store.due("mutable", refresh=True))
        store.fetch("missing", lambda: pd.DataFrame())
        store.state["missing"]["updated_at"] = (utc_now() - timedelta(days=8)).isoformat()
        self.assertTrue(store.due("missing"))

    def test_access_pause_persists_and_expires(self):
        access_pause(self.config)
        access_pause(self.config, AccessBlocked("FBref refused access"))
        with self.assertRaises(AccessBlocked):
            access_pause(self.config)
        path = Path(self.temp.name) / "access_pause.json"
        state = json.loads(path.read_text())
        state["retry_after"] = (utc_now() - timedelta(hours=1)).isoformat()
        path.write_text(json.dumps(state))
        access_pause(self.config)

    def test_lock_released_on_interruption(self):
        root = Path(self.temp.name)
        with self.assertRaises(KeyboardInterrupt):
            with output_lock(root):
                with self.assertRaises(RuntimeError):
                    with output_lock(root):
                        pass
                raise KeyboardInterrupt
        self.assertFalse((root / ".download.lock").exists())

    def test_multilevel_columns_are_preserved(self):
        frame = pd.DataFrame([[2, 10]], columns=pd.MultiIndex.from_tuples(
            [("performance", "goals"), ("passing", "total")]), index=pd.Index(["Player"], name="player"))
        path = Path(self.temp.name) / "multi.csv"
        export_frame(path, frame)
        pd.testing.assert_frame_equal(frame, pd.read_csv(path, header=[0, 1], index_col=0))
        self.assertEqual(json.loads(path.with_suffix(".schema.json").read_text())["column_levels"], 2)

    def test_discovery_and_second_run_skips_matches(self):
        self.config["leagues"] = ["ENG-Premier League"]
        factory = lambda *args: FakeReader()
        run_once(self.config, factory=factory)
        first_calls = len(FakeReader.calls)
        self.assertEqual(first_calls, 3)
        run_once(self.config, factory=factory)
        self.assertEqual(len(FakeReader.calls), first_calls)
        root = Path(self.temp.name) / "ENG-Premier%0020League" / "2010-2011"
        report = json.loads((root / "coverage.json").read_text())
        self.assertEqual(report["match_ids_selected"], 1)
        self.assertEqual(report["schedule_rows_without_report"], 1)
        self.assertFalse((root.parent / "2026-2027").exists())

    def test_player_failure_does_not_skip_other_match_tables(self):
        reader = FakeReader()
        def fail(*args, **kwargs):
            raise ValueError("unavailable")
        reader.read_player_match_stats = fail
        # Keep lineup independent of the failed player method.
        reader.read_lineup = lambda **kwargs: pd.DataFrame({"x": [1, 2]}, index=pd.Index(["home", "away"], name="team"))
        root = Path(self.temp.name) / "season"
        collect_season(reader, root, self.config, 2010)
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["matches/abc12345/player_summary"]["status"], "error")
        self.assertEqual(manifest["matches/abc12345/lineup"]["status"], "ok")
        self.assertEqual(manifest["matches/abc12345/events"]["status"], "ok")

    @unittest.skipUnless(importlib.util.find_spec("soccerdata"), "Install scraping extra for integration check")
    def test_installed_soccerdata_discovery_and_access_guard(self):
        # Use the real library parser, with local HTML and a fake browser only.
        os.environ.setdefault("SOCCERDATA_DIR", str(ROOT / "data/raw/fbref/soccerdata_runtime"))
        import soccerdata as sd
        driver = Mock()
        with patch.object(sd.FBref, "_init_webdriver", return_value=driver):
            reader = make_reader(self.config, ["ENG-Premier League"], ["2010-2011", "2026-2027"])
        catalog = b'''<table id="seasons"><thead><tr><th>Season</th></tr></thead><tbody>
          <tr><th data-stat="year_id"><a href="/2010">2010-2011</a></th></tr>
          <tr><th data-stat="year_id"><a href="/2024">2024-2025</a></th></tr>
          </tbody></table>'''
        leagues = pd.DataFrame({"url": ["/league"]}, index=pd.Index(reader.leagues, name="league"))
        with patch.object(reader, "read_leagues", return_value=leagues), patch.object(reader, "get", return_value=BytesIO(catalog)):
            discovered = reader.discover_seasons()
        self.assertEqual(list(discovered.index.get_level_values("season")), ["1011", "2425"])
        self.assertEqual(reader.seasons, ["1011", "2627"])
        driver.page_source = "<html><body>Verify you are human</body></html>"
        with patch("soccerdata._common.time.sleep"), self.assertRaises(AccessBlocked):
            reader._download_and_save("https://fbref.com/en/comps/")
        self.assertEqual(driver.get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
