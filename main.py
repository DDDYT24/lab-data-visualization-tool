import argparse
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.empty:
            print(f"[Error] File loaded but is empty: {path}")
            sys.exit(1)
        return df
    except FileNotFoundError:
        print(f"[Error] File not found: {path}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to read '{path}': {e}")
        sys.exit(1)


def clean_data(df: pd.DataFrame, dropna: bool, select_cols=None) -> pd.DataFrame:
    if select_cols:
        missing = [c for c in select_cols if c not in df.columns]
        if missing:
            print(f"[Error] Columns not found: {missing}")
            sys.exit(1)
        df = df[select_cols].copy()
    if dropna:
        before = len(df)
        df = df.dropna().copy()
        after = len(df)
        print(f"[Info] Drop NA: {before - after} rows removed (kept {after}).")
    return df


def save_data(df: pd.DataFrame, out_path: str) -> None:
    abs_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(abs_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    try:
        df.to_csv(abs_path, index=False)
        print(f"[OK] Saved cleaned data -> {abs_path}")
    except Exception as e:
        print(f"[Error] Failed to save to '{out_path}': {e}")
        sys.exit(1)


def plot_data(df: pd.DataFrame, kind: str, x: str, ys: list[str],
              save_path: str, title: str | None,
              dpi: int = 120, size: tuple[float, float] = (6, 4)) -> None:
    if x not in df.columns:
        print(f"[Error] X column not found: '{x}'")
        sys.exit(1)
    for y in ys:
        if y not in df.columns:
            print(f"[Error] Y column not found: '{y}'")
            sys.exit(1)

    df = df.copy()

    # 时间列智能识别
    try:
        df[x] = pd.to_datetime(df[x])
    except Exception:
        pass

    plt.figure(figsize=size, dpi=dpi)
    for y in ys:
        if kind == "line":
            plt.plot(df[x], df[y], label=y)
        elif kind == "scatter":
            plt.scatter(df[x], df[y], label=y)
        elif kind == "bar":
            plt.bar(df[x], df[y], label=y)
        else:
            print("[Error] Unknown plot kind. Use: line | scatter | bar")
            sys.exit(1)

    plt.xlabel(x)
    plt.ylabel(", ".join(ys))
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()

    abs_path = os.path.abspath(save_path)
    try:
        plt.savefig(abs_path)
        print(f"[OK] Saved plot -> {abs_path}")
    except Exception as e:
        print(f"[Error] Failed to save plot: {e}")
        sys.exit(1)
    finally:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Lab Data Visualization Tool — Day 5"
    )
    subparsers = parser.add_subparsers(dest="command")

    # clean 子命令
    clean_p = subparsers.add_parser("clean", help="Clean data and save CSV")
    clean_p.add_argument("-i", "--input", default="data.csv")
    clean_p.add_argument("-o", "--output", default="cleaned.csv")
    clean_p.add_argument("--dropna", action="store_true")
    clean_p.add_argument("--cols", nargs="*")

    # plot 子命令
    plot_p = subparsers.add_parser("plot", help="Plot data to image")
    plot_p.add_argument("-i", "--input", default="cleaned.csv")
    plot_p.add_argument("--kind", choices=["line", "scatter", "bar"], default="line")
    plot_p.add_argument("--x", required=True)
    plot_p.add_argument("--y", nargs="+", required=True)  # 多 y
    plot_p.add_argument("--save", default="plot.png")
    plot_p.add_argument("--title")
    plot_p.add_argument("--dpi", type=int, default=120)
    plot_p.add_argument("--size", nargs=2, type=float, metavar=("W", "H"))

    args = parser.parse_args()

    if args.command == "clean":
        df = load_data(args.input)
        df_clean = clean_data(df, dropna=args.dropna, select_cols=args.cols)
        save_data(df_clean, args.output)

    elif args.command == "plot":
        df = load_data(args.input)
        w, h = (args.size if args.size else (6, 4))
        plot_data(df, args.kind, args.x, args.y,
                  args.save, args.title, dpi=args.dpi, size=(w, h))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
