"""Check actual downloaded PNG pixels and PDF vector intervals against API bins.

Uses Pillow already provided by the API environment. The PDF reader here targets the
Flate-compressed vector page content produced by LabViz's Matplotlib PDF backend.
"""

import json
from pathlib import Path
import re
import sys
import zlib

from PIL import Image

root = Path(sys.argv[1])
bins = json.loads(sys.argv[2])
count = max(b["count"] for b in bins)
span = bins[-1]["end"] - bins[0]["start"]

# A0/A1 also describe axes and text; the long filled staircase is the histogram.
pdf = (root / "export.pdf").read_bytes()
paths = []
number = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf, re.S):
    try:
        content = zlib.decompress(match[1]).decode("latin1")
    except zlib.error:
        continue
    if not content.startswith("/DeviceRGB CS"):
        continue
    for drawing in re.findall(rf"((?:{number}\s+{number}\s+[ml]\s*)+)\s*f", content):
        coordinates = [float(v) for v in re.findall(number, drawing)]
        if len(coordinates) == 4 * len(bins) + 4:
            paths.append(list(zip(coordinates[::2], coordinates[1::2])))
assert len(paths) == 1, f"Expected one histogram staircase, got {len(paths)}"
points = paths[0]
left, right = min(p[0] for p in points), max(p[0] for p in points)
baseline = min(p[1] for p in points)
height_unit = (max(p[1] for p in points) - baseline) / count
for index, b in enumerate(bins):
    start, end = points[1 + 2 * index : 3 + 2 * index]
    assert (
        abs((start[0] - left) / (right - left) - (b["start"] - bins[0]["start"]) / span)
        < 1e-6
    )
    assert (
        abs((end[0] - left) / (right - left) - (b["end"] - bins[0]["start"]) / span)
        < 1e-6
    )
    assert abs((start[1] - baseline) / height_unit - b["count"]) < 1e-5
assert b"/ca 0.55" in pdf

# Locate red (#dc2626, opacity .55 on white) intervals in the raster itself.
image = Image.open(root / "export.png").convert("RGB")
pixels = image.load()


def is_red(x, y):
    r, g, b = pixels[x, y]
    return abs(r - 236) <= 2 and abs(g - 136) <= 2 and abs(b - 136) <= 2


rows = []
for y in range(image.height // 2, image.height):
    xs = [x for x in range(image.width) if is_red(x, y)]
    if len(xs) > image.width * 0.35:
        rows.append((y, xs))
assert rows, "No histogram raster baseline found"
bottom, xs = rows[-1]
left, right = min(xs), max(xs) + 1
heights = []
for b in bins:
    center = ((b["start"] + b["end"]) / 2 - bins[0]["start"]) / span
    x = round(left + center * (right - left))
    ys = [y for y in range(bottom + 1) if is_red(x, y)]
    if not b["count"]:
        assert not ys or bottom - ys[-1] > 6
        heights.append(0)
        continue
    assert ys and bottom - ys[-1] <= 3
    top = ys[-1]
    for y in reversed(ys[:-1]):
        if top - y > 6:
            break
        top = y
    heights.append(bottom - top + 1)
unit = max(heights) / count
for height, b in zip(heights, bins):
    assert abs(height / unit - b["count"]) < 0.08, (height / unit, b["count"])

(root / "artifact-content-check.json").write_text(
    json.dumps(
        {
            "status": "PASS",
            "bins": bins,
            "pngPixelHeights": heights,
            "pdfVectorIntervals": "match",
            "svgIntervals": "checked by Playwright",
        },
        indent=2,
    ),
    encoding="utf-8",
)
