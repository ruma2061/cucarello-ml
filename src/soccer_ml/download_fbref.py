"""Resumable SoccerData FBref exports. Run with --help for overrides."""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
import logging
import math
import os
from pathlib import Path
import re
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "fbref.json"
LOG = logging.getLogger(__name__)


class AccessBlocked(RuntimeError):
    """Stop the run when the provider refuses access."""


class DownloadRefused(BaseException):
    """Escape SoccerData's internal Exception retry loop on a refusal."""


def utc_now():
    return datetime.now(timezone.utc)


def atomic_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def export_frame(path, frame):
    """Preserve all returned values, indices, and multi-level column headings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=True, encoding="utf-8")
    temporary.replace(path)
    atomic_json(path.with_suffix(".schema.json"), {
        "index_names": list(frame.index.names),
        "column_names": list(frame.columns.names),
        "column_levels": frame.columns.nlevels,
        "columns": [list(c) if isinstance(c, tuple) else c for c in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "rows": len(frame),
    })


def safe_name(value):
    # Retain ordinary names, escape unsafe characters without collisions.
    return re.sub(r'[^A-Za-z0-9_.-]', lambda m: f"%{ord(m[0]):04X}", str(value))


def season_start(code):
    """Handle SoccerData's 1011 codes, full ranges, and calendar-year seasons."""
    text = str(code)
    if re.fullmatch(r"\d{4}[-/]\d{2,4}", text):
        return int(text[:4])
    if re.fullmatch(r"(?:19|20)\d{2}", text) and text != "2021":
        return int(text)
    if re.fullmatch(r"\d{4}", text):
        start = int(text[:2])
        if int(text[2:]) == (start + 1) % 100:
            return 2000 + start if start < 70 else 1900 + start
    raise ValueError(f"Unrecognized season code: {code}")


class Store:
    """Per-season checkpoints; an interrupted export never becomes complete."""

    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.path = root / "manifest.json"
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def due(self, key, refresh=False):
        entry = self.state.get(key)
        if self.config["force"] or not entry:
            return True
        status = entry["status"]
        age = (utc_now() - datetime.fromisoformat(entry["updated_at"])).total_seconds() / 3600
        if status in {"error", "partial"}:
            return age >= self.config["retry_error_hours"]
        if status == "empty":
            return age >= self.config["retry_missing_hours"]
        if not (self.root / f"{key}.csv").exists() or not (self.root / f"{key}.schema.json").exists():
            return True
        return refresh and age >= self.config["refresh_hours"]

    def fetch(self, key, function, refresh=False, check=None):
        if not self.due(key, refresh):
            return None
        LOG.info("Downloading %s / %s", self.root, key)
        for attempt in range(self.config["retries"] + 1):
            try:
                frame = function()
                if not isinstance(frame, pd.DataFrame):
                    raise TypeError("Expected a pandas DataFrame from SoccerData")
                export_frame(self.root / f"{key}.csv", frame)
                status = "empty" if frame.empty else "ok"
                if status == "ok" and check is not None and not check(frame):
                    status = "partial"
                self.record(key, status, rows=len(frame))
                return frame
            except AccessBlocked as error:
                self.record(key, "error", error=str(error))
                raise
            except Exception as error:
                LOG.warning("%s: %s: %s", key, type(error).__name__, error)
                if attempt < self.config["retries"]:
                    time.sleep(self.config["retry_backoff"] * 2 ** attempt)
                else:
                    self.record(key, "error", error=f"{type(error).__name__}: {error}")
        return None

    def record(self, key, status, **details):
        self.state[key] = {"status": status, "updated_at": utc_now().isoformat(), **details}
        atomic_json(self.path, self.state)


def make_reader(config, leagues, seasons):
    """Keep the library's HTML cache and restrict refreshes to mutable pages."""
    os.environ.setdefault("SOCCERDATA_DIR", str(Path(config["output_dir"]) / "soccerdata_runtime"))
    import soccerdata as sd
    class RecordingFBref(sd.FBref):
        @classmethod
        def _all_leagues(cls):
            # SoccerData keys its provider registry by concrete class name.
            return sd.FBref._all_leagues().copy()

        def discover_seasons(self):
            # SoccerData otherwise defaults to five seasons and its .loc selector
            # can raise if any requested season is absent. Select all source rows
            # first, then filter the returned catalog in run_once.
            original = self._season_ids
            self._discovering = True
            self._season_ids = slice(None)
            try:
                return super().read_seasons()
            finally:
                self._season_ids = original
                self._discovering = False

        def get(self, url, filepath=None, max_age=None, no_cache=False, var=None):
            # Season/league indexes can change, and unfinished seasons need fresh aggregates.
            is_match = "/en/matches/" in url
            age = None if is_match else config["cache_max_age_days"]
            refresh = no_cache
            if filepath and not is_match:
                mutable = Path(filepath).name == "leagues.html" or Path(filepath).name.startswith("seasons_")
                if not getattr(self, "_discovering", False):
                    mutable = mutable or any(not self._is_complete(l, s) for l in self.leagues for s in self.seasons)
                if mutable:
                    age = config["refresh_hours"] / 24
                    # SoccerData's max_age accepts integer days, so check sub-day ages here.
                    refresh = refresh or (Path(filepath).exists() and
                        time.time() - Path(filepath).stat().st_mtime >= age * 86400)
                    age = None
            if config["force"]:
                refresh = True
            cached = Path(filepath).exists() if filepath else False
            result = super().get(url, filepath, max_age=age, no_cache=refresh, var=var)
            self._request_log.parent.mkdir(parents=True, exist_ok=True)
            with self._request_log.open("a", encoding="utf-8") as log:
                log.write(json.dumps({"at": utc_now().isoformat(), "url": url,
                    "cache_file": str(filepath), "cache_present_before": cached,
                    "refresh_requested": refresh}) + "\n")
            return result

        def _is_captcha_present(self):
            page = self._driver.page_source.lower()
            if super()._is_captcha_present() or any(marker in page for marker in (
                "too many requests", "access denied", "403 forbidden", "request blocked",
            )):
                raise DownloadRefused("FBref refused access or displayed a browser challenge")
            return False

        def solve_captcha(self):
            raise DownloadRefused("FBref displayed a browser challenge")

        def _download_and_save(self, url, filepath=None, var=None):
            # Escape internal library retries on an explicit refusal; no challenge solving.
            try:
                return super()._download_and_save(url, filepath, var)
            except DownloadRefused as error:
                raise AccessBlocked(f"{error}: {url}") from error

    reader = RecordingFBref(leagues=leagues, seasons=seasons,
        data_dir=Path(config["output_dir"]) / "html_cache", headless=config["headless"],
        path_to_browser=Path(config["browser_path"]) if config["browser_path"] else None)
    reader.rate_limit = config["request_delay"]
    reader.max_delay = config["request_jitter"]
    reader._request_log = Path(config["output_dir"]) / "requests.jsonl"
    return reader


def close_reader(reader):
    driver = getattr(reader, "_driver", None)
    if driver is not None:
        try:
            driver.quit()
        except Exception:
            LOG.warning("Could not close browser cleanly", exc_info=True)


def access_pause(config, error=None):
    """Persist a conservative pause so immediate restarts do not repeat a block."""
    path = Path(config["output_dir"]) / "access_pause.json"
    if error is not None:
        until = utc_now() + timedelta(hours=config["block_cooldown_hours"])
        atomic_json(path, {"error": str(error), "recorded_at": utc_now().isoformat(),
                           "retry_after": until.isoformat()})
    elif path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        until = datetime.fromisoformat(state["retry_after"])
        if utc_now() < until:
            remaining = (until - utc_now()).total_seconds() / 3600
            raise AccessBlocked(f"Collection paused after a previous access refusal. Wait {remaining:.1f} more hours. "
                                f"Details: {path}")


@contextmanager
def output_lock(root):
    """Prevent two collectors from racing on the same checkpoint files."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".download.lock"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(f"Collector lock exists: {path}. If no collector is running, remove it.") from error
    try:
        with handle:
            handle.write(str(utc_now()))
        yield
    finally:
        path.unlink(missing_ok=True)


def both_teams(frame):
    if "team" not in frame.index.names:
        return False
    return frame.index.get_level_values("team").nunique() == 2


def collect_season(reader, root, config, year):
    store = Store(root, config)
    # Always refresh the schedule through SoccerData's cache, then reuse that
    # exact schedule during per-match calls to avoid thousands of repeat fetches.
    schedule = store.fetch("schedule", lambda: reader.read_schedule(), refresh=True)
    if schedule is None:
        if store.state.get("schedule", {}).get("status") != "ok":
            return
        schedule = pd.read_csv(root / "schedule.csv", index_col=[0, 1, 2])
    if schedule.empty:
        return
    reader.read_schedule = lambda force_cache=False: schedule
    mutable = not all(reader._is_complete(l, s) for l in reader.leagues for s in reader.seasons)
    for stat in config["season_stats"]:
        store.fetch(f"player_season/{stat}", lambda s=stat: reader.read_player_season_stats(s), mutable)
        for opponent in ([False, True] if config["opponent_stats"] else [False]):
            side = "against" if opponent else "for"
            store.fetch(f"team_season/{side}/{stat}",
                lambda s=stat, o=opponent: reader.read_team_season_stats(s, opponent_stats=o), mutable)
    for stat in config["team_match_stats"]:
        for opponent in ([False, True] if config["opponent_stats"] else [False]):
            side = "against" if opponent else "for"
            store.fetch(f"team_match/{side}/{stat}",
                lambda s=stat, o=opponent: reader.read_team_match_stats(s, opponent_stats=o, force_cache=True), mutable)
    if "game_id" not in schedule or "match_report" not in schedule:
        store.record("match_discovery", "error", error="Schedule missing game_id or match_report")
        return
    reports = schedule.dropna(subset=["game_id", "match_report"])
    ids = list(dict.fromkeys(reports["game_id"].astype(str)))
    if config["max_matches"] is not None:
        ids = ids[:config["max_matches"]]
    for match_id in ids:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", match_id):
            store.record("match_discovery", "error", error=f"Invalid match ID: {match_id}")
            continue
        keys = [f"matches/{match_id}/player_{stat}" for stat in config["player_match_stats"]]
        keys += [f"matches/{match_id}/{kind}" for kind in ("lineup", "events")
                 if config["lineups" if kind == "lineup" else "events"]]
        cache_path = Path(config["output_dir"]) / "html_cache" / f"match_{match_id}.html"
        if (cache_path.exists() and time.time() - cache_path.stat().st_mtime >= config["retry_missing_hours"] * 3600
                and any(store.state.get(key, {}).get("status") in {"partial", "empty", "error"}
                        and store.due(key) for key in keys)):
            # A report may acquire tables after it first appears; don't retry stale HTML.
            cache_path.unlink(missing_ok=True)
        for stat in config["player_match_stats"]:
            key = f"matches/{match_id}/player_{stat}"
            store.fetch(key, lambda s=stat, m=match_id: reader.read_player_match_stats(s, match_id=m, force_cache=True), check=both_teams)
        if config["lineups"]:
            store.fetch(f"matches/{match_id}/lineup",
                lambda m=match_id: reader.read_lineup(match_id=m, force_cache=True), check=both_teams)
        if config["events"]:
            store.fetch(f"matches/{match_id}/events",
                lambda m=match_id: reader.read_events(match_id=m, force_cache=True))
    atomic_json(root / "coverage.json", {
        "updated_at": utc_now().isoformat(), "season_start_year": year,
        "schedule_rows": len(schedule), "reports_available": len(reports),
        "match_ids_selected": len(ids), "limited_run": config["max_matches"] is not None,
        "schedule_rows_without_report": int(schedule["match_report"].isna().sum()),
        "statuses": dict(Counter(value["status"] for value in store.state.values())),
        "note": "ok means returned nonempty, not that every possible field exists; player tables require both teams.",
    })
    LOG.info("%s: %s", root, dict(Counter(value["status"] for value in store.state.values())))


def run_once(config, factory=make_reader):
    root = Path(config["output_dir"])
    end = config["end_year"] if config["end_year"] is not None else datetime.now().year
    requested = [f"{y}-{y+1}" for y in range(config["start_year"], end + 1)]
    for league in config["leagues"]:
        league_root = root / safe_name(league)
        reader = factory(config, [league], requested)
        try:
            discovery = Store(league_root / "catalog", config)
            discovery.fetch("leagues", reader.read_leagues, refresh=True)
            seasons = discovery.fetch("seasons", reader.discover_seasons, refresh=True)
            if seasons is None and discovery.state.get("seasons", {}).get("status") == "ok":
                seasons = pd.read_csv(league_root / "catalog/seasons.csv", index_col=[0, 1])
            if seasons is None or seasons.empty:
                continue
            # Use the source-returned season codes/URLs rather than inventing match URLs.
            available = list(dict.fromkeys(seasons.index.get_level_values("season").astype(str)))
            selected = [(s, season_start(s)) for s in available
                        if config["start_year"] <= season_start(s) <= end]
        finally:
            close_reader(reader)
        for season, year in sorted(selected, key=lambda pair: pair[1]):
            reader = factory(config, [league], [season])
            try:
                collect_season(reader, league_root / f"{year}-{year+1}", config, year)
            finally:
                close_reader(reader)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--leagues", nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-matches", type=int, help="Limit per league-season for a pilot")
    parser.add_argument("--watch", action="store_true", default=None)
    parser.add_argument("--force", action="store_true", default=None)
    parser.add_argument("--poll-hours", type=float)
    parser.add_argument("--dry-run", action="store_true", help="Show configuration without network access")
    args = parser.parse_args(argv)
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    overrides = json.loads(args.config.read_text(encoding="utf-8"))
    unknown = set(overrides) - set(config)
    if unknown:
        parser.error(f"Unknown config settings: {sorted(unknown)}")
    config.update(overrides)
    for key in ("start_year", "end_year", "leagues", "output_dir", "max_matches", "watch", "force", "poll_hours"):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    config["output_dir"] = str((ROOT / config["output_dir"]).resolve())
    end = config["end_year"] if config["end_year"] is not None else datetime.now().year
    if not 1900 <= config["start_year"] <= end <= datetime.now().year:
        parser.error("Years must satisfy 1900 <= start_year <= end_year <= current year")
    for key in ("request_delay", "request_jitter", "retry_backoff", "refresh_hours", "retry_missing_hours", "poll_hours", "retry_error_hours", "block_cooldown_hours"):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] < 0:
            parser.error(f"{key} must be a finite nonnegative number")
    if config["poll_hours"] <= 0 or config["refresh_hours"] <= 0 or config["retry_missing_hours"] <= 0:
        parser.error("Poll/refresh/missing intervals must be positive")
    if config["retry_error_hours"] <= 0 or config["block_cooldown_hours"] <= 0:
        parser.error("Error retry and block cooldown intervals must be positive")
    if config["request_delay"] < 7:
        parser.error("request_delay must be at least 7 seconds")
    if not isinstance(config["retries"], int) or config["retries"] < 0:
        parser.error("retries must be a nonnegative integer")
    if not isinstance(config["cache_max_age_days"], int) or config["cache_max_age_days"] < 1:
        parser.error("cache_max_age_days must be a positive integer")
    if config["max_matches"] is not None and (not isinstance(config["max_matches"], int) or config["max_matches"] < 1):
        parser.error("max_matches must be a positive integer")
    if not config["leagues"] or "Big 5 European Leagues Combined" in config["leagues"]:
        parser.error("Select individual leagues for per-match discovery")
    for key, allowed in {
        "season_stats": {"standard", "keeper", "shooting", "playing_time", "misc"},
        "team_match_stats": {"schedule", "keeper", "shooting", "misc"},
        "player_match_stats": {"summary", "keepers"},
    }.items():
        if not isinstance(config[key], list) or set(config[key]) - allowed:
            parser.error(f"{key} must be a list drawn from {sorted(allowed)}")
    for key in ("opponent_stats", "lineups", "events", "force", "watch", "headless"):
        if not isinstance(config[key], bool):
            parser.error(f"{key} must be true or false")
    return args, config


def main(argv=None):
    args, config = parse_args(argv)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(config["output_dir"])
    try:
        access_pause(config)
        os.environ.setdefault("SOCCERDATA_DIR", str(root / "soccerdata_runtime"))
        import soccerdata as sd
        unknown = set(config["leagues"]) - set(sd.FBref.available_leagues())
        if unknown:
            raise ValueError(f"Unsupported leagues: {sorted(unknown)}")
        with output_lock(root):
            atomic_json(root / "run_config.json", {**config, "soccerdata_version": version("soccerdata"), "started_at": utc_now().isoformat()})
            while True:
                try:
                    run_once(config)
                except AccessBlocked as error:
                    access_pause(config, error)
                    raise
                if not config["watch"]:
                    break
                LOG.info("Collection pass finished. Waiting %s hours.", config["poll_hours"])
                time.sleep(config["poll_hours"] * 3600)
    except ImportError:
        LOG.error('Install scraping dependencies: python -m pip install -e ".[scraping]"')
        return 1
    except KeyboardInterrupt:
        LOG.info("Stopped. Re-run the same command to resume.")
        return 130
    except (AccessBlocked, RuntimeError, ValueError) as error:
        atomic_json(root / "last_error.json", {"at": utc_now().isoformat(), "type": type(error).__name__, "error": str(error)})
        LOG.error("Stopped: %s", error)
        return 1
    except Exception as error:
        atomic_json(root / "last_error.json", {"at": utc_now().isoformat(), "type": type(error).__name__, "error": str(error)})
        LOG.exception("Stopped unexpectedly; details saved in %s", root / "last_error.json")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
