# FBref downloader

This collector uses SoccerData's FBref web scraper, not an official FBref API.
It defaults to the five project leagues, starting with **2010–2011**, through the
latest season FBref lists within the current year. Years are inclusive **season
start years**, not calendar-date cutoffs. A null end year advances automatically.
The collector discovers published seasons first, rather than assuming every
requested season exists. Discovery CSVs retain FBref's season URLs.

## Install and run

From the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[scraping]"
.\.venv\Scripts\python.exe scripts/download_fbref.py --dry-run
.\.venv\Scripts\python.exe scripts/download_fbref.py
```

Chrome/Chromium and a compatible browser driver are required by SoccerData.
SoccerData may download its driver on first use. Set `browser_path` in the config
if Chrome is not found. No API key is required.

For a smaller initial run (season and team tables are still collected):

```powershell
.\.venv\Scripts\python.exe scripts/download_fbref.py --leagues "ENG-Premier League" --start-year 2024 --end-year 2024 --max-matches 2
```

For a collector that keeps running and checks for new data daily:

```powershell
.\.venv\Scripts\python.exe scripts/download_fbref.py --watch --poll-hours 24
```

Press Ctrl+C to stop. Repeat the command to resume. Continuous mode is a foreground
process; it runs while the computer and process remain running. No scheduled task
or background service is installed. Use one collector per output directory.

## Configuration

Edit `config/fbref.json`, or supply a JSON override file with `--config PATH`.
Relative `output_dir` values resolve against the project folder. CLI options
override JSON. All collection settings are in JSON:

| Setting | Purpose |
| --- | --- |
| `start_year`, `end_year` | Inclusive season start years; null end follows the current year |
| `leagues` | Individual SoccerData league IDs; default is the project five |
| `season_stats` | All supported player/team season categories by default |
| `team_match_stats` | All supported team match-log categories by default |
| `player_match_stats` | Both currently supported player match categories: summary and keepers |
| `opponent_stats` | Also collect team opponent statistics |
| `lineups`, `events` | Collect lineups and goal/card/substitution event tables |
| `request_delay`, `request_jitter` | Default 20 seconds plus up to 10 seconds extra per uncached request; minimum delay 7 seconds |
| `retries`, `retry_backoff` | Additional retries per failed table, with exponential waits |
| `retry_error_hours` | Default 24 hours before another attempt at failed/partial exports |
| `block_cooldown_hours` | Default 24-hour pause after a detected access refusal, preserved across restarts |
| `refresh_hours` | Revisit catalogs, schedules, and unfinished-season aggregates |
| `retry_missing_hours` | Recheck empty tables after this interval |
| `cache_max_age_days` | Cache lifetime for historical non-match pages |
| `force` | Rebuild exports and bypass HTML cache; expensive, normally false |
| `watch`, `poll_hours` | Repeat collection after each pass finishes and the interval elapses |
| `max_matches` | Pilot limit per league-season; null means every discovered report |
| `browser_path`, `headless` | SoccerData browser settings |

## Output

```text
data/raw/fbref/
  run_config.json                  Effective settings and SoccerData version
  requests.jsonl                   Requested URLs, cache locations, and timestamps
  html_cache/                      Source HTML cached by SoccerData
  soccerdata_runtime/              SoccerData logs/config (unless SOCCERDATA_DIR is set)
  ENG-Premier%0020League/
    catalog/                       League/season discovery CSVs and manifest
    2010-2011/
      schedule.csv
      player_season/               Every configured season category
      team_season/for/             Team season statistics
      team_season/against/         Opponent season statistics
      team_match/for/              Team match logs
      team_match/against/          Opponent match logs
      matches/<FBref-match-ID>/
        player_summary.csv
        player_keepers.csv
        lineup.csv
        events.csv
      manifest.json                Per-table status, timestamp, row count, errors
      coverage.json                Match-report coverage and status counts
```

Every CSV has a `.schema.json` companion describing index names, column levels,
data types, and row counts. All SoccerData-returned columns are retained without
feature selection or imputation. This is a raw ingestion layer: SoccerData itself
parses and normalizes source HTML. Original downloaded HTML is also retained.
CSV tables with multi-level headings have multiple header rows; use the schema's
`column_levels` and `index_names` when loading them with pandas. For example:

```python
schema = json.loads(path.with_suffix(".schema.json").read_text())
frame = pd.read_csv(path,
    header=list(range(schema["column_levels"])),
    index_col=list(range(len(schema["index_names"]))))
```

The collector visits every published match report in each selected league-season.
Matches without report links remain visible in the schedule and coverage summary.
Tables are checkpointed individually; failures don't discard other match tables.
An `ok` player match table contains both teams, but does not prove every possible
player or field was available. `partial`, `empty`, and `error` explicitly identify
incomplete results. Empty tables use `retry_missing_hours` before the next attempt.
Failed/partial tables use `retry_error_hours`.
Successful historical match tables are reused; use
`force` if you specifically need to retrieve later corrections.

The HTML cache prevents fetching the same report for each separate table. Successful
exports are replaced atomically, with checkpoints written afterwards. A lock
prevents concurrent runs; after a hard crash, remove `.download.lock` only after
confirming no collector is running. Don't edit manifests while a collector runs.

## Availability and limits

- FBref removed its provider's advanced data in January 2026, including historical
  data. No script can retrieve removed xG/passing/possession tables from FBref.
- Basic fields, report availability, and player detail vary by season and league.
  The supported SoccerData tables are attempted independently; unavailable tables
  are logged rather than represented as complete.
- This downloads everything exposed by the configured SoccerData FBref methods,
  not every unrelated FBref page. Extra HTML tables are retained in the source cache.
  It does not supply wages, market values, or a complete biographical database.
- Access challenges/explicit refusals stop collection, preserving checkpoints.
  The script doesn't rotate proxies or automatically solve access challenges.
- Detected access refusals also write `access_pause.json` and prevent another run
  until the configured cooldown expires. `last_error.json` records the latest
  fatal error. A cooldown is a conservative local policy, not a guarantee that
  FBref will accept the next attempt. Parser errors don't establish throttling.
- Many thousands of reports can take days. Keep conservative delays and check
  provider access/reuse terms before running or sharing a dataset.
- Website changes can break SoccerData parsers. Inspect manifests after each run;
  completion of a pass does not mean all requested data was available.

References:
[SoccerData FBref methods](https://soccerdata.readthedocs.io/en/stable/reference/fbref.html),
[FBref data removal announcement](https://www.sports-reference.com/blog/2026/01/fbref-stathead-data-update/).

Offline verification: `python -m unittest discover -s tests -v` after installing the project.
