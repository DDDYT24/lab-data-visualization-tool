"""Stage private D1 production inputs without creating a release acceptance receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "b587db848ba5f1930657d5a316e0160c71864ead"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(ROOT / "outputs/desktop-d01") and not output.exists()
    runtime = args.runtime_payload.resolve()
    spec = importlib.util.spec_from_file_location("stage_candidate", ROOT / "V2.0/packaging/windows-v23/stage_candidate.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    old_manifest = stage.read(runtime / "payload-manifest.json")
    runtime_files = {name: value for name, value in old_manifest["files"].items() if name.startswith("runtime/")}
    for name, expected in runtime_files.items():
        assert digest(runtime / name) == expected, name
    output.mkdir(parents=True)
    stage.copy_tree(runtime / "runtime", output / "runtime")
    dist = ROOT / "V2.0/web/.next-v23-package"
    stage.copy_tree(dist / "standalone", output / "V2.0")
    stage.copy_tree(dist / "static", output / "V2.0/web/.next-v23-package/static")
    stage.copy_tree(ROOT / "V2.0/web/public", output / "V2.0/web/public")
    stage.portable_web_config(output / "V2.0/web")
    for entry in ("labviz_api", "migrations", "samples/v22"):
        stage.copy_tree(ROOT / "V2.0/api" / entry, output / "V2.0/api" / entry)
    for name in ("requirements.txt", "requirements.lock.txt", "pyproject.toml", "alembic.ini"):
        shutil.copyfile(ROOT / "V2.0/api" / name, output / "V2.0/api" / name)
    (output / "V2.0/contracts").mkdir()
    for source in (ROOT / "V2.0/contracts").glob("*.json"):
        shutil.copyfile(source, output / "V2.0/contracts" / source.name)
    stage.copy_tree(ROOT / "V2.0/assets", output / "V2.0/assets")
    stage.copy_tree(ROOT / "V2.0/packaging/windows-v23/bin", output / "bin")
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(ROOT / name, output / name)
    archive = subprocess.check_output(["git", "archive", "--format=zip", BASE, "--", "V2.0/api", "V2.0/web", "V2.0/contracts", "V2.0/assets", "V2.0/packaging/windows-v23"], cwd=ROOT)
    source_hashes = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as sources:
        for member in sources.infolist():
            if member.is_dir():
                continue
            name = member.filename
            actual = (ROOT / name).read_bytes()
            expected = sources.read(member)
            if actual != expected:
                assert b"\0" not in actual and actual.replace(b"\r\n", b"\n") == expected.replace(b"\r\n", b"\n"), "Product source differs from the frozen RC: " + name
            source_hashes[name] = hashlib.sha256(actual).hexdigest()
    files = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    candidate = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    metadata = {"format": "labviz-desktop-comparison-payload-v1", "appId": stage.APP_ID, "candidateId": candidate, "dataEpoch": 1, "sourceCommit": BASE, "scope": "Private D1 comparison; no release or installer acceptance receipt", "webBuildId": (dist / "BUILD_ID").read_text().strip(), "runtimeFileCount": len(runtime_files), "runtimeSourceCandidate": old_manifest["candidateId"], "sourceFilesSha256": source_hashes}
    (output / "candidate-build.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "payload-manifest.json").write_text(json.dumps({"files": files, "candidateId": candidate, "sourceCommit": BASE}, indent=2), encoding="utf-8")
    print(json.dumps({"candidateId": candidate, "files": len(files), "bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file())}))


if __name__ == "__main__":
    main()
