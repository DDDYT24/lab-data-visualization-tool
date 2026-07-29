"""PostgreSQL engine, session, transaction, and health-check helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker


@dataclass(frozen=True)
class DatabaseHealth:
    """Bounded readiness result that does not leak credentials or SQL details."""

    ready: bool
    detail: str


class Database:
    """Own the production PostgreSQL engine and short-lived ORM sessions."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        parsed = make_url(url)
        if parsed.get_backend_name() != "postgresql":
            raise ValueError("The production database URL must use PostgreSQL.")
        self.engine: Engine = create_engine(
            url,
            echo=echo,
            pool_pre_ping=True,
            pool_recycle=1_800,
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
            autoflush=False,
        )

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Commit one unit of work, rolling it back on every failure."""

        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def health(self) -> DatabaseHealth:
        """Check connectivity with a minimal query and a stable result."""

        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError:
            return DatabaseHealth(ready=False, detail="postgres-unavailable")
        return DatabaseHealth(ready=True, detail="ok")

    def dispose(self) -> None:
        """Close pooled connections during shutdown or tests."""

        self.engine.dispose()
