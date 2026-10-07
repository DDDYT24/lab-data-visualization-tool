"""Render the synthetic About-page figures through the same code as API exports."""

from __future__ import annotations

import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from labviz_api.models import ChartSpec  # noqa: E402
from labviz_api.processing import default_chart_spec, load_dataframe, render_chart  # noqa: E402
from labviz_api.sample_catalog import sample_payload  # noqa: E402


def main() -> None:
    destination = API_ROOT.parent / "web" / "public" / "about"
    destination.mkdir(parents=True, exist_ok=True)
    examples = (
        ("time-series", "line", "Synthetic response over time", "response-2d.svg"),
        ("surface-3d", "surface3d", "Synthetic X/Y/Z response surface", "surface-3d.svg"),
    )
    for slug, chart_type, title, filename in examples:
        sample, payload = sample_payload(slug)
        if not sample.synthetic:
            raise RuntimeError(f"About figure source '{slug}' is not marked synthetic.")
        frame, _, _, _ = load_dataframe(
            payload,
            sample.filename,
            requested_sheet_name=sample.sheet_name,
            header_row=sample.header_row,
        )
        specification = default_chart_spec(frame, chart_type)
        specification["title"] = title
        specification["export"]["format"] = "svg"
        specification["export"]["sizePreset"] = "double-column"
        payload_svg = render_chart(frame, ChartSpec.model_validate(specification))
        target = destination / filename
        target.write_bytes(payload_svg)
        print(f"{target.name}: {len(payload_svg)} bytes; source={sample.slug}; format=svg")


if __name__ == "__main__":
    main()
