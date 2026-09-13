"""Generate the Windows icon from the reusable LabViz PNG source."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets" / "labviz-logo.png"
TARGET = ROOT / "assets" / "labviz-logo.ico"
SIZES = (16, 24, 32, 48, 64, 128, 256)


def main() -> None:
    with Image.open(SOURCE) as source:
        image = source.convert("RGBA")
        image.save(TARGET, format="ICO", sizes=[(size, size) for size in SIZES])
    print(f"Generated {TARGET} with sizes: {', '.join(str(size) for size in SIZES)}")


if __name__ == "__main__":
    main()
