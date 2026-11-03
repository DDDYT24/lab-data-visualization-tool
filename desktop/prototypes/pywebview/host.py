"""Windowed pywebview comparison host; close hides without destroying state."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sys.executable).with_name("launcher.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trial = Path(config["trialRoot"])

    def event(name: str) -> None:
        with (trial / "host-events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": name, "utcSeconds": time.time()}) + "\n")

    event("host-start")
    os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = f"--remote-debugging-port={config['debugPort']} --remote-debugging-address=127.0.0.1"
    controller = subprocess.Popen([config["python"], config["controller"], str(config_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    response = json.loads(controller.stdout.readline())
    event("services-ready")
    import webview
    import pystray
    from PIL import Image

    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.settings["REMOTE_DEBUGGING_PORT"] = config["debugPort"]
    window = webview.create_window("LabViz D1 pywebview", response["url"], width=1280, height=900)
    quitting = False

    def closing() -> bool | None:
        if not quitting:
            window.hide()
            event("hidden-to-tray")
            return False
        return None

    def show(_icon=None, _item=None) -> None:
        window.show()
        window.restore()
        event("restored-from-tray")

    def leave(_icon=None, _item=None) -> None:
        nonlocal quitting
        quitting = True
        event("explicit-exit")
        icon.stop()
        window.destroy()

    icon = pystray.Icon("LabVizD1pywebview", Image.open(config["icon"]), "LabViz D1 pywebview", pystray.Menu(pystray.MenuItem("Open LabViz", show, default=True), pystray.MenuItem("Exit LabViz", leave)))
    window.events.closing += closing
    window.events.loaded += lambda: event("page-load")
    window.events.shown += lambda: event("window-created")
    icon.run_detached()

    def commands() -> None:
        last = None
        while not quitting:
            try:
                command = json.loads((trial / "host-command.json").read_text(encoding="utf-8"))
                if command.get("trialId") == config["trialId"] and command["id"] != last:
                    last = command["id"]
                    if command["action"] == "hide":
                        window.hide()
                        event("benchmark-hidden")
                    elif command["action"] == "show":
                        show()
                    elif command["action"] == "exit":
                        leave()
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.2)

    threading.Thread(target=commands, daemon=True).start()
    try:
        webview.start(gui="edgechromium", debug=False, private_mode=False, storage_path=config["webviewProfile"])
    finally:
        icon.stop()
        if controller.poll() is None:
            controller.stdin.write("stop\n")
            controller.stdin.flush()
            controller.wait(timeout=100)


if __name__ == "__main__":
    main()
