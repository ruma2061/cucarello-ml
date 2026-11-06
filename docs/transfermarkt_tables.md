# Transfermarkt tables for notebooks

Build the tables from the project folder:

```powershell
python scripts/prepare_transfermarkt.py
# Restrict leagues or inclusive season-start years if needed:
python scripts/prepare_transfermarkt.py --start-year 2015 --end-year 2023 --competitions GB1 L1
```

The script reads local CSVs; it downloads nothing and does not alter raw files.
It finds flat CSVs or nested `players.csv/players.csv` folders automatically.
Input/output paths and CSV reading chunk size can be changed with `--help`.
Defaults select GB1 (England), FR1 (France), IT1 (Italy), L1 (Germany), ES1 (Spain)
from the 2010 season onward. This local snapshot actually begins in 2012 for these
leagues. No rows are fabricated for earlier seasons.

Exports live under `data/processed/transfermarkt/`. Every table is available as
CSV and as a table inside `transfermarkt.sqlite`; SQLite uses Python's built-in
sqlite3 module and is useful for selecting a small subset without loading all rows.
Reruns replace generated exports. Do not rebuild while a notebook holds the database
open. A failed rebuild can leave a mixture of old/new CSVs; rerun successfully
before analysis. The SQLite database is replaced only after all table writes finish.

| Table | One row represents |
| --- | --- |
| `player_match` | One recorded player appearance in a game, enriched with match-time club, opponent, outcome, profile, lineup, prior performance, and historical valuation |
| `player_season` | One player / league / season, combining clubs after transfers within that league |
| `player_club_season` | One player / club / league / season, keeping transfers separate |
| `team_match` | One club's perspective on a match |
| `matches` | One selected match, including both teams and final score |
| `lineups` | One recorded lineup player per match, including unused substitutes where present |
| `players`, `clubs`, `competitions` | Full source reference tables, including entities outside the selected leagues |
| `player_valuations` | Valuation history for players with selected appearance records; dates outside the selected seasons are retained |

`coverage.csv` reports observed player/match coverage by league and season.
`build_metadata.json` records source paths/sizes, date ranges, counts, missing
competitions, and games without appearance data. `data_dictionary.csv` lists every
table's columns, inferred types, missing counts, and modeling role. Inspect these
files before training. Missing source statistics are retained, not imputed as zero.
Aggregate sums use observed values; missing-count columns help identify incomplete
season totals. Appearances do not include players who never appeared; use `lineups`
for unused substitutes. Missing next-season records remain null, not zero.

## Loading

Open `notebooks/transfermarkt_modeling.ipynb` for a working example, or:

```python
import sqlite3
import pandas as pd
from pathlib import Path

root = Path(r"C:\Projects\soccerML")  # change for another computer
with sqlite3.connect(root / "data/processed/transfermarkt/transfermarkt.sqlite") as con:
    players = pd.read_sql_query("SELECT * FROM player_season", con)
    matches = pd.read_sql_query(
        "SELECT * FROM player_match WHERE competition_id = ? AND season_start_year = ?",
        con, params=("GB1", 2023), parse_dates=["date"])
```

## Choosing features and outcomes

- For predicting a player's next match: `prior_appearances`, `prior_goals`,
  `prior_assists`, `prior_minutes_played`, `prior_goals_per90`, and
  `prior_assists_per90` use earlier appearances in the same league/season, across
  clubs. The first appearance has zero prior totals and undefined per-90 rates.
  Dates, not future fixtures, define ordering; data assumes a player does not have
  two appearances on the same day within a league.
- `market_value_before_match_eur` uses the latest valuation strictly before the
  match date. Its date and age in days are included. No future/same-day valuation
  is filled backwards. Publication timestamps are unavailable, so this is an
  observation-date rule, not a guarantee of real-time availability.
- Season tables similarly use `market_value_before_first_match_eur` before each
  player's first observed appearance. That is not necessarily the league opener.
- `target_scored` is a current-game classification outcome. Current-game goals,
  minutes, assists, cards, lineup details, and final score must not be used as
  predictors for a pre-match model. `match_result` is W/D/L from the player's club.
- `target_next_season_*` columns on `player_season` describe the exact next season
  **within the same league**. Missing records may reflect transfer, retirement,
  missing data, or incomplete coverage. Training on observed outcomes therefore
  selects a subset of players. Later-season totals may themselves be incomplete.
- `position`, `sub_position`, `height_cm`, and `foot` are snapshot profile values;
  historical position changes and height growth aren't reconstructed. Age is
  computed from birth date at the match (or first season appearance).
- No salary or weight is fabricated. Latest career valuations, international caps,
  and current club membership remain in reference tables only; don't join those
  snapshot fields into historical models without considering leakage.
- Use explicit feature lists and chronological validation. Never select every
  numeric column automatically. Exclude all `target_*` columns from features.

Events and transfer-fee histories are not flattened into the modeling tables;
they remain available in the raw folder for later task-specific features.
