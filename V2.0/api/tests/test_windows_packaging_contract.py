from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGING_ROOT = REPO_ROOT / "V2.0" / "packaging" / "windows"
MANIFEST_PATH = PACKAGING_ROOT / "package-manifest.json"


def test_windows_package_contract_is_local_first_and_honest_about_release_status() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["contractVersion"] == "v1"
    assert manifest["releaseLine"] == "V2.2"
    assert manifest["platform"] == "windows-x64"
    assert manifest["localFirst"] is True
    assert manifest["buildStatus"] == "test-installer-source"
    assert manifest["installer"]["artifact"] is None
    assert manifest["installer"]["cleanMachineVerified"] is False
    assert manifest["runtime"]["runtimeDependencyInstall"] is False
    assert manifest["runtime"]["networkRequiredAfterInstall"] is False
    assert manifest["layout"]["bindAddress"] == "127.0.0.1"
    assert manifest["upgradePolicy"]["uninstall"].lower().find("silently") >= 0

    included = set(manifest["include"])
    assert "runtime/python" in included
    assert "runtime/node" in included
    assert "V2.0/web/server.js" in included
    assert "V2.0/web/node_modules" in included
    assert "V2.0/api/samples/v22" in included
    assert "V2.0/assets/labviz-logo.ico" in included
    assert "V2.0/assets/labviz-logo.svg" in included
    assert "V2.0/assets/fonts" in included
    fonts = REPO_ROOT / "V2.0" / "assets" / "fonts"
    assert (fonts / "NotoSansSC-VF.ttf").is_file()
    assert "SIL OPEN FONT LICENSE" in (fonts / "OFL.txt").read_text(encoding="utf-8")
    excluded = "\n".join(manifest["exclude"]).lower()
    for prohibited in (".env", ".labviz", ".venv", "node_modules", "raw user", "credentials"):
        assert prohibited in excluded


def test_windows_packaging_scripts_protect_the_data_boundary() -> None:
    validator = (PACKAGING_ROOT / "validate-package.ps1").read_text(encoding="utf-8")
    launcher = (PACKAGING_ROOT / "bin" / "start-labviz-portable.ps1").read_text(encoding="utf-8")
    documentation = (PACKAGING_ROOT / "README.md").read_text(encoding="utf-8")

    for prohibited in (".env", ".labviz", ".venv", "node_modules", "outputs", "test-results"):
        assert prohibited in validator
    assert "RequireBundledRuntimes" in validator
    assert "node_modules outside traced production web runtime" in validator
    assert "stage-candidate.ps1" in documentation
    assert "--require-hashes" in documentation
    assert "development virtual environment directly" in documentation
    staging = (PACKAGING_ROOT / "stage-candidate.ps1").read_text(encoding="utf-8")
    assert 'Where-Object { $_.Name -ine "site-packages" }' in staging
    assert "Python site-packages contains unrelated node_modules" in staging
    assert "OutputRoot must be new or empty" in staging
    assert "%LOCALAPPDATA%" in documentation
    assert "P5-2" in documentation
    assert "remains open" in documentation
    assert "LABVIZ_DATABASE_PATH" in launcher
    assert "LABVIZ_OBJECT_STORAGE_ROOT" in launcher
    assert "127.0.0.1" in launcher
    assert "health check" in launcher
    assert "SkipOpenBrowser" in launcher
    assert "quotedWebServer" in launcher
    assert (PACKAGING_ROOT / "LabViz.iss").is_file()
    installer_source = (PACKAGING_ROOT / "LabViz.iss").read_text(encoding="utf-8")
    assert "PrivilegesRequired=lowest" in installer_source
    assert "start-labviz-installed.ps1" in installer_source
    assert (PACKAGING_ROOT / "test-installer.ps1").is_file()
    assert "Silent uninstall did not preserve local data" in (
        PACKAGING_ROOT / "test-installer.ps1"
    ).read_text(encoding="utf-8")


def test_reusable_labviz_icon_sources_are_present() -> None:
    assert (REPO_ROOT / "V2.0" / "assets" / "labviz-logo.svg").is_file()
    assert (REPO_ROOT / "V2.0" / "assets" / "labviz-logo.png").is_file()
    assert (REPO_ROOT / "V2.0" / "assets" / "labviz-logo.ico").is_file()
