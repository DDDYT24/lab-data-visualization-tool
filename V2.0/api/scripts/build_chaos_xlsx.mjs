// Copy into a generated support/ directory and link its node_modules to the
// bundled workspace dependencies before running. The sibling data/ directory
// receives four synthetic, multi-sheet XLSX fixtures.
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const supportDir = path.join(root, "support");
// artifact-tool writes inspect sidecars alongside workbook exports. Preserve
// those diagnostics under support/ so data/ contains upload fixtures only.
const inspectionRunDir = path.join(supportDir, `inspection-${Date.now()}`);
await fs.mkdir(inspectionRunDir);
const specs = JSON.parse(await fs.readFile(path.join(supportDir, "xlsx-specs.json"), "utf8"));
if (specs.length !== 4) throw new Error(`Expected four workbook specifications, got ${specs.length}`);
for (const spec of specs) {
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("Measurements");
  sheet.getRange("A1").write([spec.headers, ...spec.rows]);
  sheet.getRange("A1:D1").format.font = { name: "Arial", size: 10, bold: true };
  sheet.getRange("A1:D1").format.fill = "#DBEAFE";
  sheet.showGridLines = false;
  const metadata = workbook.worksheets.add("Metadata");
  metadata.getRange("A1:B2").values = [
    ["Origin", "Deterministic synthetic LabViz fixture"],
    ["Purpose", "Test import, quality, surface validation and exports"],
  ];
  metadata.showGridLines = false;
  workbook.recalculate();
  const check = await workbook.inspect({
    kind: "table", range: "Measurements!A1:D3", include: "values,formulas",
    tableMaxRows: 3, tableMaxCols: 4, maxChars: 1500,
  });
  if (!check.ndjson.includes(spec.headers[0])) throw new Error(`Inspection failed for ${spec.filename}`);
  const errors = await workbook.inspect({
    kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 50 }, maxChars: 1000,
  });
  if (errors.ndjson.includes('"cell"')) throw new Error(`Formula error found in ${spec.filename}`);
  const preview = await workbook.render({ sheetName: "Measurements", range: "A1:D8", scale: 1, format: "png" });
  await fs.writeFile(path.join(supportDir, `${spec.filename}.png`),
    new Uint8Array(await preview.arrayBuffer()));
  const blob = await SpreadsheetFile.exportXlsx(workbook);
  await blob.save(path.join(root, "data", spec.filename));
  await fs.rename(
    path.join(root, "data", `${spec.filename}.inspect.ndjson`),
    path.join(inspectionRunDir, `${spec.filename}.inspect.ndjson`),
  );
  console.log(`Built ${spec.filename}`);
}
