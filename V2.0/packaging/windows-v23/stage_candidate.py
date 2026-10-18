"""Stage a self-contained V2.3 payload after verified identity/data contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

APP_ID = "55B46BB7-101E-46DC-951A-3DE7E402EDD0"
ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checked(path: Path) -> Path:
    path = path.absolute()
    for member in (path, *path.parents):
        if member.is_symlink() or member.is_junction():
            raise ValueError("Package sources/outputs cannot use links or junctions.")
    return path.resolve()


def source_inputs() -> dict:
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            env=env,
        )
        .decode("utf-8")
        .split("\0")
    )
    members = sorted({name for name in names if name and (ROOT / name).is_file()})
    with ThreadPoolExecutor(max_workers=16) as pool:
        files = dict(pool.map(lambda name: (name, sha(ROOT / name)), members))
    return {
        "sourceCommit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, env=env, text=True
        ).strip(),
        "sourceDirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=ROOT,
                env=env,
            ).strip()
        ),
        "filesSha256": files,
    }


def receipt(path: Path) -> None:
    record = read(path)
    if record.get("appId") != APP_ID or not record.get("identityAndDataContractPassed"):
        raise ValueError(
            "V2.3 identity/data verification receipt is required before staging."
        )
    for name, digest in record["sourceSha256"].items():
        source = checked(ROOT / name)
        if not source.is_relative_to(ROOT) or sha(source) != digest:
            raise ValueError(
                "Identity/data implementation changed after verification: " + name
            )
    for evidence in record["evidence"]:
        source = checked(Path(evidence["path"]))
        if sha(source) != evidence["sha256"]:
            raise ValueError("Identity/data evidence changed.")
    suite = ET.parse(record["pythonTests"]).getroot()
    if len(list(suite.iter("testcase"))) < 29 or any(
        list(suite.iter(x)) for x in ("failure", "error", "skipped")
    ):
        raise ValueError("Identity/data API tests have not passed.")
    native = read(Path(record["nativeTests"]))
    if not native.get("passed") or not all(native["checks"].values()):
        raise ValueError("Native Windows isolation contracts have not passed.")


def copy_tree(source: Path, destination: Path, *, skip: set[str] | None = None) -> None:
    source = checked(source)
    for directory, folders, files in os.walk(source, followlinks=False):
        for entry in folders + files:
            member = Path(directory) / entry
            if member.is_symlink() or member.is_junction():
                raise ValueError("Package trees cannot contain links or junctions.")
        folders[:] = [
            n for n in folders if n not in (skip or set()) and n != "__pycache__"
        ]
        target = destination / Path(directory).relative_to(source)
        target.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name.endswith((".pyc", ".pyo")) or name in (skip or set()):
                continue
            shutil.copyfile(Path(directory) / name, target / name)


def portable_web_config(web: Path) -> None:
    server = web / "server.js"
    text = server.read_text(encoding="utf-8")
    match = re.search(r"^const nextConfig = (.+)$", text, re.MULTILINE)
    if match is None:
        raise ValueError("Unknown standalone server configuration format.")
    config = json.loads(match[1])
    config.pop("outputFileTracingRoot", None)
    config.pop("repoRoot", None)
    if "turbopack" in config:
        config["turbopack"].pop("root", None)
    replacement = "const nextConfig = " + json.dumps(config, separators=(",", ":"))
    replacement += "\nnextConfig.outputFileTracingRoot = path.resolve(__dirname, '..')"
    replacement += "\nnextConfig.repoRoot = __dirname"
    replacement += "\nnextConfig.turbopack.root = path.resolve(__dirname, '..')"
    text = text[: match.start()] + replacement + text[match.end() :]
    guard = """
const proxyTarget = new URL(process.env.LABVIZ_API_PROXY_TARGET || 'http://invalid')
if (process.env.LABVIZ_INSTALL_LINE !== '2.3' ||
    !/^[a-f0-9]{32}$/.test(process.env.LABVIZ_PROFILE_ID || '') ||
    !(process.env.LABVIZ_LOCAL_ACCESS_KEY || '').length ||
    proxyTarget.protocol !== 'http:' || proxyTarget.hostname !== '127.0.0.1' || !proxyTarget.port ||
    proxyTarget.pathname !== '/' || proxyTarget.search || proxyTarget.hash || proxyTarget.username || proxyTarget.password) {
  throw new Error('Start LabViz V2.3 using its installed launcher and owned profile.')
}
const reserved = new Set([3000,8000,3001,8001,3305,8305])
if (reserved.has(Number(process.env.PORT)) || reserved.has(Number(proxyTarget.port))) {
  throw new Error('A protected V2.2/source/acceptance port cannot be used.')
}
"""
    text = guard + text.replace(
        "process.env.HOSTNAME || '0.0.0.0'", "'127.0.0.1'"
    ).replace(
        "parseInt(process.env.PORT, 10) || 3000",
        "parseInt(process.env.PORT, 10) || 3301",
    )
    server.write_text(text, encoding="utf-8")
    required = web / ".next-v23-package/required-server-files.json"
    document = read(required)
    document["appDir"] = "."
    document["config"] = {**config, "repoRoot": ".", "outputFileTracingRoot": ".."}
    document["config"]["turbopack"] = {**config.get("turbopack", {}), "root": ".."}
    required.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")

    # Source maps are diagnostics, not runtime imports; remove local build-machine paths.
    def normalized(value):
        if isinstance(value, str):
            return value.replace(str(ROOT), "/labviz-source").replace(
                str(ROOT).replace("\\", "/"), "/labviz-source"
            )
        if isinstance(value, dict):
            return {k: normalized(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalized(v) for v in value]
        return value

    for source_map in web.rglob("*.map"):
        document = json.loads(source_map.read_text(encoding="utf-8"))
        source_map.write_text(
            json.dumps(normalized(document), separators=(",", ":")), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-runtime", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--web-licenses", type=Path, required=True)
    args = parser.parse_args()
    receipt(args.receipt)
    inputs = source_inputs()
    output = checked(args.output)
    if (
        output.exists()
        or output.is_relative_to(ROOT / "V2.0")
        or output.is_relative_to(ROOT / "V1.1")
    ):
        raise ValueError("Staging output must be new and outside source trees.")
    if not output.is_relative_to(ROOT / "outputs/v23-release"):
        raise ValueError("Use this workspace's private V2.3 release output directory.")
    web = ROOT / "V2.0/web"
    dist = web / ".next-v23-package"
    standalone = checked(dist / "standalone")
    if not (standalone / "web/server.js").is_file():
        raise ValueError(
            "Build the isolated .next-v23-package standalone server first."
        )
    output.mkdir(parents=True)
    copy_tree(standalone, output / "V2.0")
    copy_tree(dist / "static", output / "V2.0/web/.next-v23-package/static")
    copy_tree(web / "public", output / "V2.0/web/public")
    portable_web_config(output / "V2.0/web")
    for entry in ("labviz_api", "migrations", "samples/v22"):
        copy_tree(ROOT / "V2.0/api" / entry, output / "V2.0/api" / entry)
    for entry in (
        "requirements.txt",
        "requirements.lock.txt",
        "pyproject.toml",
        "alembic.ini",
    ):
        shutil.copyfile(ROOT / "V2.0/api" / entry, output / "V2.0/api" / entry)
    # Runtime preparation imports the workflow contract from this sibling directory.
    # Ship the versioned contracts, excluding test fixtures and development outputs.
    (output / "V2.0/contracts").mkdir(parents=True)
    for contract in sorted((ROOT / "V2.0/contracts").glob("*.json")):
        shutil.copyfile(checked(contract), output / "V2.0/contracts" / contract.name)
    copy_tree(ROOT / "V2.0/assets", output / "V2.0/assets")
    copy_tree(SOURCE / "bin", output / "bin")
    shutil.copyfile(SOURCE / "identity.json", output / "identity.json")
    shutil.copyfile(ROOT / "LICENSE", output / "LICENSE")
    shutil.copyfile(ROOT / "THIRD_PARTY_NOTICES.md", output / "THIRD_PARTY_NOTICES.md")
    (output / "licenses").mkdir()
    shutil.copyfile(
        checked(args.web_licenses), output / "licenses/WEB-THIRD-PARTY-LICENSES.txt"
    )
    copy_tree(args.python_runtime, output / "runtime/python")
    exclusions = {
        p.name
        for p in args.site_packages.iterdir()
        if p.name == "pip" or p.name.startswith("pip-")
    }
    copy_tree(
        args.site_packages, output / "runtime/python/Lib/site-packages", skip=exclusions
    )
    (output / "runtime/python/python313._pth").write_text(
        "python313.zip\n.\nLib/site-packages\n../../V2.0/api\nimport site\n",
        encoding="ascii",
    )
    (output / "runtime/node").mkdir(parents=True)
    shutil.copyfile(checked(args.node), output / "runtime/node/node.exe")
    for license_file in (args.node.parent / "LICENSE", args.node.parent / "README.md"):
        if license_file.exists():
            shutil.copyfile(license_file, output / "runtime/node" / license_file.name)
    for p in output.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(output)
        if any(
            n
            in (
                ".env",
                ".labviz",
                ".venv",
                "outputs",
                "test-results",
                "playwright-report",
                ".git",
                "__pycache__",
            )
            or n.startswith(".env.")
            for n in rel.parts
        ):
            raise ValueError("Prohibited runtime/user path: " + str(rel))
        if "node_modules" in rel.parts and not str(rel).replace("\\", "/").startswith(
            ("V2.0/web/", "V2.0/node_modules/")
        ):
            raise ValueError("Untraced node_modules outside the production web tree.")
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "MPLCONFIGDIR": str(output.parent / ".stage-font-cache"),
    }
    python = output / "runtime/python/python.exe"
    version = subprocess.check_output(
        [str(python), "-B", "-c", "import sys;print(sys.version.split()[0])"],
        env=env,
        text=True,
    ).strip()
    if not version.startswith("3.13."):
        raise ValueError("The release runtime must be Python 3.13.")
    subprocess.run(
        [
            str(python),
            "-B",
            "-X",
            "utf8",
            "-c",
            "import fastapi,matplotlib,numpy,pandas,scipy,pyarrow,uvicorn; from labviz_api import research_preparation, research_routes, sample_catalog; from labviz_api.installed_profile import APP_ID; [sample_catalog.sample_payload(s.slug) for s in sample_catalog.public_samples()]; print('Bundled API resources and imports passed:',APP_ID)",
        ],
        cwd=output,
        env=env,
        check=True,
    )
    node_version = subprocess.check_output(
        [str(output / "runtime/node/node.exe"), "--version"], text=True
    ).strip()
    if source_inputs() != inputs:
        raise ValueError(
            "Source changed during staging; preserve this result and build a new candidate."
        )
    (output / "source-inputs.json").write_text(
        json.dumps(inputs, indent=2), encoding="utf-8"
    )
    members = [p for p in sorted(output.rglob("*")) if p.is_file()]
    with ThreadPoolExecutor(max_workers=16) as pool:
        files = dict(
            pool.map(lambda p: (p.relative_to(output).as_posix(), sha(p)), members)
        )
    if max(len(name) for name in files) > 121:
        raise ValueError("Payload exceeds the installer's verified Windows path bound.")
    identity = {
        "appId": APP_ID,
        "version": "2.3.0-rc.1",
        "dataEpoch": 1,
        "files": files,
    }
    candidate_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output / "payload-manifest.json").write_text(
        json.dumps({**identity, "candidateId": candidate_id}, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "appId": APP_ID,
        "candidateId": candidate_id,
        "version": "2.3.0-rc.1",
        "dataEpoch": 1,
        "webBuildId": (output / "V2.0/web/.next-v23-package/BUILD_ID")
        .read_text()
        .strip(),
        "pythonVersion": version,
        "nodeVersion": node_version,
        "runtimeDependencyInstall": False,
        "networkRequiredAfterInstall": False,
        "identityReceiptSha256": sha(args.receipt),
        "sourceCommit": inputs["sourceCommit"],
        "sourceDirty": inputs["sourceDirty"],
        "maximumPayloadRelativePath": max(len(name) for name in files),
    }
    (output / "candidate-build.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
