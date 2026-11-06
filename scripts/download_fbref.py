"""Allow running directly from the checkout without installing this project."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_ml.download_fbref import main

if __name__ == "__main__":
    raise SystemExit(main())
