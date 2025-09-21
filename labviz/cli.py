from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# 无显示环境下也能渲染图片
matplotlib.use("Agg")


# ---------------- Core helpers ----------------
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Failed to read CSV: {path}") from e


def validate_columns_exist(df: pd.DataFrame, required: Iterable[str]) -> None:
    req = list(required)
    missing = [c for c in req if c not in df.columns]
    if missing:
        have = list(map(str, df.columns.tolist()))
        raise ValueError(f"Missing columns: {missing}; available: {have}")


def clean_dataframe(
    df: pd.DataFrame,
    dropna: bool = True,
    method: Optional[str] = None,  # "ffill" | "bfill" | None
) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates()
    if method is not None:
        if method not in {"ffill", "bfill"}:
            raise ValueError("method must be one of: ffill, bfill, None")
        if method == "ffill":
            out = out.ffill()
        else:
            out = out.bfill()
    if dropna:
        out = out.dropna()
    return out


def plot_df(
    df: pd.DataFrame,
    x: str,
    ys: List[str],
    kind: str,
    outpath: Path,
    show: bool = False,
) -> Path:
    kind = kind.lower()
    if kind not in {"line", "scatter", "bar"}:
        raise ValueError("kind must be one of: line, scatter, bar")

    fig, ax = plt.subplots(figsize=(7, 4))
    for y in ys:
        if kind == "line":
            ax.plot(df[x], df[y], label=y)
        elif kind == "scatter":
            ax.scatter(df[x], df[y], label=y)
        else:  # bar
            ax.bar(df[x], df[y], label=y)

    ax.set_xlabel(x)
    ax.set_ylabel(", ".join(ys))
    ax.legend()
    fig.tight_layout()

    # 自动补 .png 扩展名
    if outpath.suffix == "":
        outpath = outpath.with_suffix(".png")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return outpath


# ---------------- CLI ----------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="labviz",
        description="Clean, validate and visualize lab CSV data.",
    )
    p.add_argument("-i", "--input", type=Path, required=True, help="Input CSV path")
    p.add_argument("-x", "--x", type=str, required=True, help="X-axis column")
    p.add_argument("-y", "--y", type=str, nargs="+", required=True, help="Y-axis column(s)")
    p.add_argument("--type", choices=["line", "scatter", "bar"], default="line", help="Plot type")
    p.add_argument("-o", "--out", type=Path, default=Path("plot.png"), help="Output image path")
    p.add_argument(
        "--method",
        choices=["ffill", "bfill"],
        default=None,
        help="Fill method before dropping NaN (optional).",
    )
    p.add_argument(
        "--dropna",
        action="store_true",
        help="Drop rows with NaN after fill (default off).",
    )
    p.add_argument(
        "--show", action="store_true", help="Show plot window (if environment supports)."
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    df = load_csv(args.input)
    validate_columns_exist(df, [args.x, *args.y])

    cleaned = clean_dataframe(df, dropna=args.dropna, method=args.method)
    plot_df(cleaned, x=args.x, ys=args.y, kind=args.type, outpath=args.out, show=args.show)
    print(f"[OK] Saved figure -> {args.out if args.out.suffix else args.out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
