import fs from "node:fs";
import path from "node:path";

const [webRoot, destination] = process.argv.slice(2);
if (!webRoot || !destination) {
  throw new Error("Usage: node collect-web-licenses.mjs WEB_ROOT DESTINATION");
}
const lock = JSON.parse(fs.readFileSync(path.join(webRoot, "package-lock.json"), "utf8"));
const sections = [
  "LabViz Windows release: production web dependency notices",
  "Generated from the frozen package-lock.json and installed dependency notices.",
  "Optional/platform packages may appear even when not traced into this Windows runtime.",
  "Node.js, CPython and Python wheels retain their separate bundled license files.",
];
let count = 0;
for (const [relative, info] of Object.entries(lock.packages)) {
  if (!relative || info.dev) continue;
  const root = path.join(webRoot, relative);
  if (!fs.existsSync(root)) continue;
  const notices = [];
  const visit = (folder, depth) => {
    for (const entry of fs.readdirSync(folder, { withFileTypes: true })) {
      const file = path.join(folder, entry.name);
      if (entry.isFile() && /^(licen[cs]e|copying|notice)([._-].*)?$/i.test(entry.name)) {
        notices.push([path.relative(root, file), fs.readFileSync(file, "utf8")]);
      } else if (entry.isDirectory() && depth < 2 && /^(licenses?|notices?|compiled)$/i.test(entry.name)) {
        visit(file, depth + 1);
      }
    }
  };
  visit(root, 0);
  if (notices.length === 0) continue;
  sections.push(`\n===== ${relative} (${info.version}; declared ${info.license ?? "see notice"}) =====`);
  for (const [name, text] of notices) sections.push(`\n--- ${name} ---\n${text}`);
  count += 1;
}
if (count === 0) throw new Error("No production dependency notices found.");
fs.writeFileSync(destination, `${sections.join("\n")}\n`);
console.log(`Collected production notices for ${count} installed packages.`);
