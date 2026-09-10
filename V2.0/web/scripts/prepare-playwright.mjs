import { existsSync, mkdtempSync, renameSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const projectRoot = path.resolve(import.meta.dirname, "..");
const staleDevCache = path.join(projectRoot, ".next", "dev");

if (existsSync(staleDevCache)) {
  const quarantine = mkdtempSync(path.join(tmpdir(), "labviz-next-dev-"));
  renameSync(staleDevCache, path.join(quarantine, "dev"));
  console.log(`Moved stale Next.js development cache to ${quarantine}`);
}
