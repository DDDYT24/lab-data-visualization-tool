import { spawn } from "node:child_process";
import { writeFileSync, rmSync } from "node:fs";
import { resolve } from "node:path";

// Only the dedicated Playwright browser tree; native queries avoid slow per-poll WMI.
export function sampleBrowserMemory(root: number, stopFile: string) {
  if (process.platform !== "win32" || !Number.isSafeInteger(root) || root <= 0) {
    return { stage: () => {}, stop: async () => ({ available: false, reason: "Windows process tree unavailable" }) };
  }
  rmSync(stopFile, { force: true });
  const stageFile = stopFile + "-stage";
  const child = spawn(resolve("../api/.venv/Scripts/python.exe"), [
    resolve("../api/scripts/sample_v23_browser_memory.py"), "--root", String(root), "--stop-file", stopFile,
  ], { windowsHide: true });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
  child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
  const done = new Promise<void>((finished) => {
    child.on("close", () => finished());
    child.on("error", () => finished());
  });
  return { stage: (name: string) => writeFileSync(stageFile, name), stop: async () => {
    writeFileSync(stopFile, "stop");
    const timeout = setTimeout(() => child.kill(), 8000);
    await done;
    clearTimeout(timeout);
    rmSync(stopFile, { force: true });
    rmSync(stageFile, { force: true });
    try { return JSON.parse(stdout.trim()) as Record<string, unknown>; }
    catch { return { available: false, reason: stderr.slice(0, 300) || "No sampler report" }; }
  } };
}
