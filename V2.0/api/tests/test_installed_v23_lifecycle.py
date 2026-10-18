from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import pytest

from labviz_api.installed_lifecycle import (
    backup_profile,
    import_legacy,
    recover_profile,
    remove_profile,
    replace_active,
    restore_independent,
    verify_installed,
)
from labviz_api.installed_profile import InstalledProfile, cookie_suffix
from labviz_api.source_data import file_sha


@pytest.fixture
def profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InstalledProfile:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    return InstalledProfile.open(tmp_path / "LabVizV23", create=True)


def data_at(profile: InstalledProfile) -> None:
    profile.data.mkdir()
    (profile.data / "objects").mkdir()
    (profile.data / "objects/frozen.svg").write_bytes(b"<svg>frozen scientific output</svg>")
    (profile.data / "objects/manifest.json").write_bytes(
        b"preserve object with reserved-looking name"
    )
    with closing(sqlite3.connect(profile.data / "labviz.db")) as db, db:
        db.execute("CREATE TABLE projects (id TEXT PRIMARY KEY, owner TEXT)")
        db.execute("INSERT INTO projects VALUES ('stable-id', 'unchanged-owner')")
        db.execute("CREATE TABLE research_results (revision TEXT, value REAL)")
        db.execute("INSERT INTO research_results VALUES ('frozen-revision', 1.25)")
    profile.set_epoch(1)


def rows(path: Path) -> list[tuple[str, str]]:
    with closing(sqlite3.connect(path / "labviz.db")) as db:
        return db.execute("SELECT * FROM projects ORDER BY id").fetchall()


def test_profile_identity_cookie_isolation_and_foreign_roots(
    profile: InstalledProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = InstalledProfile.open(tmp_path / "LabVizV23Profiles" / uuid4().hex, create=True)
    assert profile.cookie_suffix != other.cookie_suffix
    monkeypatch.delenv("LABVIZ_INSTALL_LINE", raising=False)
    monkeypatch.setenv("LABVIZ_DEVELOPMENT_LINE", "2.3")
    assert cookie_suffix() == "_v23"
    monkeypatch.setenv("LABVIZ_INSTALL_LINE", "2.3")
    monkeypatch.setenv("LABVIZ_PROFILE_ROOT", str(profile.root))
    monkeypatch.setenv("LABVIZ_PROFILE_ID", profile.identifier)
    assert cookie_suffix() == profile.cookie_suffix
    monkeypatch.setenv("LABVIZ_PROFILE_ID", other.identifier)
    with pytest.raises(ValueError, match="cookie identity"):
        cookie_suffix()
    for destination in (tmp_path / "LabViz", tmp_path, tmp_path / ".labviz/v23-development"):
        with pytest.raises(ValueError, match="owned"):
            InstalledProfile.open(destination, create=True)


def test_unknown_existing_root_is_not_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    target = tmp_path / "LabVizV23"
    target.mkdir()
    (target / "foreign.txt").write_text("untouched")
    with pytest.raises(ValueError, match="never adopted"):
        InstalledProfile.open(target, create=True)
    assert not (target / "profile.json").exists()
    assert (target / "foreign.txt").read_text() == "untouched"


def test_complete_backup_independent_restore_and_profile_lock(profile: InstalledProfile) -> None:
    data_at(profile)
    before = file_sha(profile.data / "labviz.db")
    with profile.lock(), pytest.raises(ValueError, match="Stop this installed profile"):
        backup_profile(profile)
    backup = backup_profile(profile)
    target = restore_independent(backup)
    assert target.identifier != profile.identifier
    assert target.cookie_suffix != profile.cookie_suffix
    assert rows(target.data) == rows(profile.data) == [("stable-id", "unchanged-owner")]
    assert file_sha(profile.data / "labviz.db") == before
    for name in ("frozen.svg", "manifest.json"):
        assert (target.data / "objects" / name).read_bytes() == (
            profile.data / "objects" / name
        ).read_bytes()
    with closing(sqlite3.connect(target.data / "labviz.db")) as db:
        assert db.execute("SELECT * FROM research_results").fetchall() == [
            ("frozen-revision", 1.25)
        ]


@pytest.mark.parametrize("damage", ["object", "epoch", "appid", "schema"])
def test_corrupt_or_foreign_backup_rejected(profile: InstalledProfile, damage: str) -> None:
    data_at(profile)
    backup = backup_profile(profile)
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    if damage == "object":
        (backup / "objects/frozen.svg").write_bytes(b"tampered")
    elif damage == "epoch":
        manifest["installed"]["dataEpoch"] = "1"
    elif damage == "appid":
        manifest["installed"]["appId"] = "foreign-app"
    else:
        manifest["installed"]["schemaSha256"] = "0" * 64
    (backup / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        restore_independent(backup)
    assert rows(profile.data) == [("stable-id", "unchanged-owner")]


def test_newer_schema_backup_is_valid_but_not_restored_by_old_runtime(
    profile: InstalledProfile,
) -> None:
    data_at(profile)
    (profile.root / "data-schema.json").write_text(
        json.dumps({"profileId": profile.identifier, "dataEpoch": 2}), encoding="utf-8"
    )
    backup = backup_profile(profile)
    assert verify_installed(backup)["installed"]["dataEpoch"] == 2
    with pytest.raises(ValueError, match="compatible runtime"):
        restore_independent(backup)
    assert profile.epoch() == 2


def test_rollback_retains_newer_data_and_recovers_interrupted_rename(
    profile: InstalledProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_at(profile)
    backup = backup_profile(profile)
    with closing(sqlite3.connect(profile.data / "labviz.db")) as db, db:
        db.execute("INSERT INTO projects VALUES ('newer-id', 'same-owner')")
    original = os.replace

    def interrupted(source: Path, destination: Path) -> None:
        if Path(source).name.startswith("restore-staging-"):
            raise OSError("injected crash between directory renames")
        original(source, destination)

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(OSError, match="injected crash"):
        replace_active(profile, backup)
    assert (profile.root / "restore-journal.json").exists()
    monkeypatch.setattr(os, "replace", original)
    recover_profile(profile)
    assert rows(profile.data) == [("stable-id", "unchanged-owner")]
    retained = next(profile.root.glob("data.retained-*"))
    assert rows(retained) == [("newer-id", "same-owner"), ("stable-id", "unchanged-owner")]
    assert not (profile.root / "restore-journal.json").exists()


def test_explicit_legacy_import_copies_without_ownership_or_source_changes(
    profile: InstalledProfile, tmp_path: Path
) -> None:
    legacy = tmp_path / "LabViz/data"
    legacy.mkdir(parents=True)
    (legacy / "objects").mkdir()
    (legacy / "objects/frozen.pdf").write_bytes(b"%PDF legacy frozen bytes")
    with closing(sqlite3.connect(legacy / "labviz-v2.db")) as db, db:
        db.execute("CREATE TABLE projects (id TEXT PRIMARY KEY, owner TEXT)")
        db.execute("INSERT INTO projects VALUES ('legacy-id', 'legacy-private-owner')")
    before = {str(p): file_sha(p) for p in legacy.rglob("*") if p.is_file()}
    import_legacy(profile, legacy)
    assert rows(profile.data) == [("legacy-id", "legacy-private-owner")]
    assert {str(p): file_sha(p) for p in legacy.rglob("*") if p.is_file()} == before
    assert (profile.data / "objects/frozen.pdf").read_bytes() == b"%PDF legacy frozen bytes"
    with pytest.raises(ValueError, match="never merges"):
        import_legacy(profile, legacy)


def test_explicit_removal_rejects_wrong_identity_and_keeps_other_profiles(
    profile: InstalledProfile, tmp_path: Path
) -> None:
    data_at(profile)
    other = restore_independent(backup_profile(profile))
    with pytest.raises(ValueError, match="exact profile ID"):
        remove_profile(profile, other.identifier)
    foreign = tmp_path / "LabViz"
    foreign.mkdir()
    (foreign / "keep").write_text("untouched")
    with pytest.raises(ValueError, match="owned"):
        remove_profile(InstalledProfile(foreign, profile.identifier), profile.identifier)
    remove_profile(profile, profile.identifier)
    assert not profile.root.exists()
    assert rows(other.data) == [("stable-id", "unchanged-owner")]
    assert (foreign / "keep").read_text() == "untouched"


def test_installed_api_cookie_authentication_tools_and_writer_lock(
    profile: InstalledProfile, tmp_path: Path
) -> None:
    from labviz_api.installed_profile import APP_ID

    program = tmp_path / "Programs/LabVizV23"
    (program / "bin").mkdir(parents=True)
    (program / "installation.json").write_text(json.dumps({"appId": APP_ID}), encoding="utf-8")
    (program / "bin/maintain-v23.ps1").write_text("# synthetic command location", encoding="utf-8")
    code = r"""
import os
from fastapi.testclient import TestClient
from labviz_api.installed_profile import environment_profile
from labviz_api.main import app, LOCAL_COOKIE
p=environment_profile()
assert LOCAL_COOKIE.endswith(p.cookie_suffix)
with TestClient(app) as c:
    assert c.get('/api/v1/local/source-data').status_code==403
    assert c.get('/api/v1/local/instance').status_code==403
    key=os.environ['LABVIZ_LOCAL_ACCESS_KEY']
    response=c.post('/api/v1/local/session',headers={'X-LabViz-Local-Key':key})
    assert response.status_code==200
    assert c.cookies.get(LOCAL_COOKIE)==key
    info=c.get('/api/v1/local/source-data').json()
    assert info['available'] and info['mode']=='installed'
    assert info['profileRoot']==str(p.root)
    assert info['tool'].endswith('maintain-v23.ps1')
    state=c.get('/api/v1/local/instance',headers={'X-LabViz-Local-Key':key}).json()
    assert state['profileId']==p.identifier and state['pid']==os.getpid()
    try: p.lock()
    except ValueError: pass
    else: raise AssertionError('Running API did not retain its writer lock')
    c.cookies.clear()
    c.cookies.set('labviz_local_session_v23',key)
    assert c.get('/api/v1/local/source-data').status_code==403
with p.lock(): pass
print('installed API identity, cookie isolation, tools, and lock release passed')
"""
    env = {
        **os.environ,
        "LABVIZ_INSTALL_LINE": "2.3",
        "LABVIZ_PROFILE_ROOT": str(profile.root),
        "LABVIZ_PROFILE_ID": profile.identifier,
        "LABVIZ_INSTALL_ROOT": str(program),
        "LABVIZ_DATABASE_PATH": str(profile.data / "labviz.db"),
        "LABVIZ_OBJECT_STORAGE_ROOT": str(profile.data / "objects"),
        "LABVIZ_LOCAL_ACCESS_KEY": "synthetic-installed-launch-key-" + "x" * 32,
        "LABVIZ_PERSISTENCE_BACKEND": "sqlite",
        "LABVIZ_OBJECT_STORAGE_BACKEND": "local",
        "LABVIZ_ENVIRONMENT": "development",
        "LABVIZ_AUTH_MODE": "console",
        "PYTHONIOENCODING": "utf-8",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
