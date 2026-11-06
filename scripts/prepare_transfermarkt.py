"""Build tables directly from a checkout, without installing the project."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soccer_ml.prepare_transfermarkt import main

if __name__ == "__main__":
    main()
