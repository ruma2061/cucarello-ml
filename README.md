# Soccer machine learning

To prepare the local Transfermarkt data for Jupyter, run
`python scripts/prepare_transfermarkt.py`, then open
`notebooks/transfermarkt_modeling.ipynb`. See [table documentation](docs/transfermarkt_tables.md).

Just the starting point for comparing supervised classifiers using soccer performance
and economic data. Initial experiments use player-season records across Ligue 1
(France), Premier League, Serie A, Bundesliga, and La Liga. Position filters allow
separate comparisons for defenders, midfielders, forwards, and goalkeepers.

## Structure

```text
data/raw/              Original source exports
data/processed/        Prepared player-season tables
src/soccer_ml/
    data.py            Load and filter datasets
    features.py        Select, impute, scale, and encode features
    models.py          Classifier registry
    evaluation.py      Classification metrics
    experiment.py      Train and compare classifiers
scripts/               Command-line entry script
models/                Future saved pipelines
reports/               Evaluation results
notebooks/             Exploratory analysis
remote/                Local remote-work files (ignored by Git)
```

## Start

From this folder, create an environment and install the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
soccer-ml data/processed/player_seasons.csv --target performance_class --test-season 2024-2025
```

Supply your prepared CSV first; no outcome labels are bundled.
See `data/README.md` for columns, units, joining guidance, and leakage precautions.
`performance_class` is an example label name, not a predefined measure of performance.

The runner compares logistic regression, decision tree, AdaBoost, random forest,
K-nearest neighbors, and SVM against a majority-class baseline. It fits preprocessing
only on training rows and tests on the requested season using earlier seasons for
training. Seasons must use consistently formatted starting years (`2023-2024`).
All available supported features are used; edit the lists in `features.py` to change them.

Add `--position DF` to compare defenders (use your dataset's position labels).
Add `--held-out-league Bundesliga` to exclude that league from training and test
on its selected season, or `--league Bundesliga` for a within-league experiment.
Use `--output reports/experiment_name.json` to keep separate results.

These are initial comparisons, not tuned models. Choose models and tune parameters
on separate validation seasons before using the final test season. Players can
appear in multiple seasons: use player-disjoint splits later if evaluating unseen
players. Next steps include source matching, a defensible outcome definition,
position-specific feature sets, feature importance, and team/match experiments.

## Download FBref data

The configurable SoccerData collector downloads published match/player tables
from the 2010–11 season onward into `data/raw/fbref/`, with checkpoints and an
optional continuous mode. See [download instructions](docs/fbref_download.md)
and edit `config/fbref.json` to change leagues, years, categories, or intervals.
Raw exports still need preparation before the model runner can use them.
