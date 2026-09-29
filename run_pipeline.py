"""Command-line entry point for the offline FairLens experiment."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fairlens.pipeline import run_full_pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        performance, fairness = run_full_pipeline()
    except Exception as exc:
        logging.exception("FairLens pipeline failed: %s", exc)
        raise SystemExit(1) from exc
    print(f"Completed: {len(performance)} performance rows and {len(fairness)} fairness group rows written to results/.")
