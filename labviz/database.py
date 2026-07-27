"""Small SQLite repository for analysis history; raw experiment data is never stored."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .core import DataSummary


@dataclass(frozen=True)
class HistoryRecord:
    id: int
    created_at: str
    filename: str
    file_hash: str
    file_size: int
    row_count: int
    column_count: int
    missing_cells: int
    duplicate_rows: int
    chart_type: str
    x_column: str | None
    y_columns: tuple[str, ...]


class HistoryStore:
    """Persist compact plot history in a single local SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS plot_history (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL CHECK (file_size >= 0),
                    row_count INTEGER NOT NULL CHECK (row_count >= 0),
                    column_count INTEGER NOT NULL CHECK (column_count >= 0),
                    missing_cells INTEGER NOT NULL CHECK (missing_cells >= 0),
                    duplicate_rows INTEGER NOT NULL CHECK (duplicate_rows >= 0),
                    chart_type TEXT NOT NULL,
                    x_column TEXT,
                    y_columns TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_plot_history_created "
                "ON plot_history(created_at DESC, id DESC)"
            )

    def record(
        self,
        *,
        filename: str,
        file_hash: str,
        file_size: int,
        summary: DataSummary,
        chart_type: str,
        x_column: str | None,
        y_columns: Sequence[str],
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO plot_history (
                    filename, file_hash, file_size, row_count, column_count,
                    missing_cells, duplicate_rows, chart_type, x_column, y_columns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    file_hash,
                    file_size,
                    summary.rows,
                    summary.columns,
                    summary.missing_cells,
                    summary.duplicate_rows,
                    chart_type,
                    x_column,
                    json.dumps(list(y_columns), ensure_ascii=False),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a history row id.")
            return cursor.lastrowid

    def recent(self, limit: int = 20) -> list[HistoryRecord]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM plot_history ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            HistoryRecord(
                id=row["id"],
                created_at=row["created_at"],
                filename=row["filename"],
                file_hash=row["file_hash"],
                file_size=row["file_size"],
                row_count=row["row_count"],
                column_count=row["column_count"],
                missing_cells=row["missing_cells"],
                duplicate_rows=row["duplicate_rows"],
                chart_type=row["chart_type"],
                x_column=row["x_column"],
                y_columns=tuple(json.loads(row["y_columns"])),
            )
            for row in rows
        ]

    def clear(self) -> int:
        with self._connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM plot_history").fetchone()[0])
            connection.execute("DELETE FROM plot_history")
        return count
