# Data

- `raw/`: untouched source exports; keep source URL, export date, units, and season notes.
- `processed/`: cleaned, joined player-season CSV files for experiments.

The initial unit of observation is one player in one team, league, and season.
Use stable IDs and an explicit player-ID mapping between providers; names alone
are not reliable join keys. Resolve transfers and duplicate season totals before training.
Use consistent season strings such as `2022-2023` so chronological sorting works.

Required columns: `player_id`, `season`, `league`, `position`, and your target column.
Optional numeric features: `age`, `height_cm`, `weight_kg`, `market_value_eur`,
`goals_per_match`, `goals_per_season`, `assists_per_match`, `minutes_played`,
`matches_won`, `annual_wage_eur`. Missing numeric values are imputed during training.
Keep currency and wage periods consistent. Record the observation date of economic data.

Define your classification target separately, with its meaning and prediction date.
For next-season performance, shift future outcomes back to the predictor-season row.
Remove any feature that reveals the label or was unavailable at prediction time.
The code only excludes the target column automatically; it cannot detect indirect leakage.
Team/match outcomes will need a separately aggregated team/match table and split strategy.

Planned sources: Transfermarkt (economic/biometric data) and FBref (performance).
Check each provider's access terms, licensing, and attribution before collection or reuse.
The FBref collector is documented in `docs/fbref_download.md`; its exports and
HTML cache live in `raw/fbref/`. It preserves all returned columns, with separate
league/season/match folders and completeness manifests. Raw collector exports
must be cleaned and joined before being used as model input.
