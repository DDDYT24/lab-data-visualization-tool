"""Command-line interface and backwards-compatible helper functions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .core import (
    clean_dataframe as clean_table,
)
from .core import (
    load_data,
    summarize_dataframe,
    validate_columns_exist,
)
from .plotting import SUPPORTED_PLOTS, create_figure, save_figure


def load_csv(path: Path) -> pd.DataFrame:
    return load_data(path)


def clean_dataframe(
    frame: pd.DataFrame,
    dropna: bool = True,
    method: str | None = None,
) -> pd.DataFrame:
    missing = method or ("drop" if dropna else "keep")
    cleaned = clean_table(frame, missing=missing)  # type: ignore[arg-type]
    return cleaned.dropna().reset_index(drop=True) if method and dropna else cleaned


def plot_df(
    frame: pd.DataFrame,
    x: str,
    ys: list[str],
    kind: str,
    outpath: Path,
    show: bool = False,
) -> Path:
    del show  # Headless rendering is deliberate and reliable in scripts and CI.
    figure = create_figure(frame, kind=kind, x=x, ys=ys)
    target = save_figure(figure, outpath)
    figure.clear()
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="labviz", description="Clean and visualize experimental tabular data."
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input data file")
    parser.add_argument("-x", "--x", help="X-axis column")
    parser.add_argument("-y", "--y", nargs="+", required=True, help="Value column(s)")
    parser.add_argument("--type", choices=SUPPORTED_PLOTS, default="line", help="Chart type")
    parser.add_argument("-o", "--out", type=Path, default=Path("plot.png"), help="PNG output")
    parser.add_argument(
        "--missing",
        choices=["keep", "drop", "ffill", "bfill", "mean", "median"],
        default="drop",
        help="Missing-value strategy",
    )
    parser.add_argument(
        "--keep-duplicates", action="store_true", help="Do not remove duplicate rows"
    )
    parser.add_argument("--title", default="", help="Optional chart title")
    parser.add_argument("--cleaned-out", type=Path, help="Also export the cleaned CSV")
    parser.add_argument("--summary", action="store_true", help="Print a JSON quality summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.type in {"line", "scatter", "bar", "surface3d"} and not args.x:
        parser.error(f"--x is required for {args.type} plots")
    frame = load_data(args.input)
    cleaned = clean_table(
        frame,
        drop_duplicates=not args.keep_duplicates,
        missing=args.missing,
    )

    required = [*args.y]
    if args.x:
        required.insert(0, args.x)
    validate_columns_exist(cleaned, required)

    figure = create_figure(
        cleaned,
        kind=args.type,
        x=args.x,
        ys=args.y,
        title=args.title,
    )
    target = save_figure(figure, args.out)
    figure.clear()

    if args.cleaned_out:
        args.cleaned_out.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(args.cleaned_out, index=False)
    if args.summary:
        print(json.dumps(summarize_dataframe(cleaned).to_dict(), indent=2))
    print(f"Saved figure: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
