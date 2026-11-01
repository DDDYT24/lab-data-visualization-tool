"""Prepare and bind a private, relocatable desktop preview folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if (ROOT/'desktop/runtime/service_runner.py').exists():
        raise ValueError('The D2 comparison is frozen. Use desktop/scripts/stage_d34.py for the current host.')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs/desktop-d02"):
        raise ValueError("Use this workspace's private desktop output directory")
    source = ROOT / "outputs/desktop-d01/payload"
    if args.prepare:
        output.mkdir(parents=True, exist_ok=False)
        for path in source.rglob("*"):
            if path.is_symlink() or path.is_junction():
                raise ValueError("Linked payload inputs are not supported")
        shutil.copytree(source, output / "app")
        print(json.dumps({"prepared": True, "output": str(output)}), flush=True)
        return
    if not (output / "app/candidate-build.json").is_file():
        raise ValueError("Prepare the common payload first")
    manifest = json.loads((source / "payload-manifest.json").read_text())
    def verify(item):
        name, expected = item
        if sha(output / "app" / name) != expected:
            raise ValueError("Copied payload differs: " + name)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify, manifest["files"].items()))
    binary = ROOT / "desktop/prototypes/tauri/target/release/LabViz.exe"
    shutil.copyfile(binary, output / "LabViz.exe")
    runtime = output / "runtime"
    runtime.mkdir(exist_ok=True)
    hashes = {}
    for name in ("controller.py", "preferences.py", "process_job.py"):
        shutil.copyfile(ROOT / "desktop/runtime" / name, runtime / name)
        hashes[name] = sha(runtime / name)
    candidate = json.loads((source / "candidate-build.json").read_text())
    record = {"format": "labviz-desktop-runtime-v1", "payload": "app", "controller": "runtime/controller.py", "candidateId": candidate["candidateId"], "controllerSha256": hashes["controller.py"], "preferencesSha256": hashes["preferences.py"], "processJobSha256": hashes["process_job.py"]}
    (output / "desktop-runtime.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (output / "README.txt").write_text("LabViz Desktop Preview\n\nDouble-click LabViz.exe. Keep this folder together.\nClosing the window keeps LabViz running in the tray. Use Exit LabViz to stop it.\nThis preview stores its own data under Local AppData\\LabVizDesktopPreview.\nThe V2.2 and V2.3 RC profiles are kept separate.\nAutomatic suspension and recovery of unsaved drafts are planned for later stages.\n", encoding="utf-8")
    receipt = {"format": "labviz-desktop-build-v1", "commonPayloadCandidate": candidate["candidateId"], "sourceCommit": candidate["sourceCommit"], "hostSha256": sha(output / "LabViz.exe"), "runtimeSha256": hashes, "verifiedCommonPayloadFiles": len(manifest["files"]), "scope": "Private D2 preview, not a signed installer or a stable release"}
    (output / "desktop-build.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
