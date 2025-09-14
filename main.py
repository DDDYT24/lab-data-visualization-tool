from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

import app


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lab Data Visualization Tool")
    p.add_argument("--file", type=Path, help="CSV file to load")
    p.add_argument("--dropna", action="store_true", help="Drop rows with NaN")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument("--version", action="version", version="labviz 0.1.0")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.info("Starting labviz…")

    if not args.file:
        logging.error("No CSV file provided")
        sys.exit(1)

    try:
        df = pd.read_csv(args.file)
        logging.info("Loaded CSV with shape %s", df.shape)
    except Exception as e:
        logging.exception("Failed to read CSV: %s", e)
        sys.exit(1)

    try:
        app.validate_columns_exist(df, ["time", "temperature"])
        cleaned = app.clean_dataframe(df, dropna=args.dropna)
    except Exception as e:
        logging.exception("Validation/cleaning failed: %s", e)
        sys.exit(1)

    print("Cleaned dataframe head:")
    print(cleaned.head())


if __name__ == "__main__":
    main()
