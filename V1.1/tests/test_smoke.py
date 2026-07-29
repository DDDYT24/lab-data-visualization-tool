from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from labviz.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_cli_end_to_end(tmp_path: Path, capsys) -> None:
    source = tmp_path / "experiment.csv"
    output = tmp_path / "chart.png"
    cleaned = tmp_path / "cleaned.csv"
    pd.DataFrame({"time": [0, 1, 2], "value": [1.0, 2.0, 3.0]}).to_csv(source, index=False)

    result = main(
        [
            "--input",
            str(source),
            "--x",
            "time",
            "--y",
            "value",
            "--out",
            str(output),
            "--cleaned-out",
            str(cleaned),
            "--summary",
        ]
    )

    assert result == 0
    assert output.exists() and cleaned.exists()
    assert '"rows": 3' in capsys.readouterr().out


def test_streamlit_app_starts_without_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LABVIZ_DB_PATH", str(tmp_path / "history.db"))
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "Lab Data Visualization"
