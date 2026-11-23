import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import ts from "typescript";
import { describe, expect, it } from "vitest";

// Charts here draw the API's RenderPlan, and the workspace shows the API's results. A statistic
// computed in the browser would be a second answer that the Matplotlib export cannot reproduce.
const FORBIDDEN: [string, RegExp][] = [
  ["quantile or percentile", /quantile|percentile/i],
  ["density estimate", /\bkde\b|kernel|gaussian|\bdensity\b|(?<!\.)\bbandwidth\b/i],
  ["spread", /variance|stdev|stddev|standardDeviation/i],
  ["sum, the start of a mean", /\.reduce\(\s*\(\s*(\w+)\b[^)]*\)\s*=>\s*(?:\{\s*return\s+)?(?:\1\s*\+|\w[\w.]*\s*\+\s*\1\b)|\bfor\s*\([^;{}]*\)\s*\{?\s*\w+\s*\+=/],
  ["binning", /histogram|binWidth|binCount|Math\.floor\(\s*\(\s*[\w.]+\s*-/i],
  ["numeric sort, the start of an order statistic", /\.(?:sort|toSorted)\(\s*\(\s*(\w+)\b[^,)]*,\s*(\w+)\b[^)]*\)\s*=>\s*(?:\{\s*return\s+)?\1[\w.!]*\s*-\s*\2\b/],
  ["power, exponential, logarithm or root", /Math\.(?:exp|log\d*|sqrt|cbrt|pow|hypot)\b|\*\*/],
];

const root = fileURLToPath(new URL("../../", import.meta.url));
const sources = ["components/common", "features/workspace", "domain"].flatMap((folder) => readdirSync(root + folder)
  .filter((name) => /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name))
  .map((name) => `${folder}/${name}`));

// Names of things the API computes and drawing arithmetic, as they read once their text is gone.
const ALLOWED: Record<string, RegExp[]> = {
  // How many y axes the panels before this one have.
  "components/common/render-plan-option.ts": [/\.reduce\(\(sum, item\) => sum \+ \(item\.axes\.y2 \? 2 : 1\), 0\)/g],
  // Where a value sits along a log axis: the same rule as plan_axes.axis_position.
  "domain/render-plan.ts": [
    /Math\.log10\((?:value|1 \+ )/g, /10 \*\* (?:position|Math\.abs\(position\))/g,
    /bandwidth: number;/g, /(?:density|histogram|quantile): string \| null;/g,
  ],
  "domain/research-charts.ts": [/bandwidth: (?:number \| null;|z\.number\(\),)/g],
  "domain/api-contract.ts": [/histograms: z\.array\(/g],
  // The order layers are drawn in.
  "domain/figure-draft.ts": [/\.sort\(\(a, b\) => a\.layer\.style\.zIndex - b\.layer\.style\.zIndex/g],
  // Whether the pointer is on a mark, and how long a retry waits.
  "domain/figure-point-hit.ts": [/\(mark\.cx \* 1000 - x\) \*\* 2 \+ \(mark\.cy \* 1000 - y\) \*\* 2 <= radius \*\* 2/g],
  "features/workspace/use-workspace-project.ts": [/500 \* 2 \*\* attempt/g],
  // The bandwidth a user types is a setting sent to the API.
  "features/workspace/research-chart-workspace.tsx": [
    /\[bandwidth, setBandwidth\]/g, /\bsecond, bandwidth, scale\b/g, /value=\{bandwidth\}/g,
    /&& bandwidth !== "" && !\(Number\(bandwidth\) > 0\);/g,
    /bandwidth: takes\(""\) && bandwidth !== "" \? Number\(bandwidth\) : null,/g,
  ],
};

/** The code without the text in it: string and template text, text between tags, comments. */
function withoutText(code: string) {
  const file = ts.createSourceFile("scanned.tsx", code, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let text = code;
  const cuts: [number, number][] = [];
  const visit = (node: ts.Node) => {
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) cuts.push([node.getStart(file) + 1, node.end - 1]);
    else if (ts.isTemplateHead(node) || ts.isTemplateMiddle(node)) cuts.push([node.getStart(file) + 1, node.end - 2]);
    else if (ts.isTemplateTail(node)) cuts.push([node.getStart(file) + 1, node.end - 1]);
    else if (ts.isJsxText(node)) cuts.push([node.pos, node.end]);
    ts.forEachChild(node, visit);
  };
  visit(file);
  for (const [start, end] of cuts.sort((a, b) => b[0] - a[0])) text = text.slice(0, start) + text.slice(end);
  return text.replace(/\/\*[\s\S]*?\*\/|(?<!:)\/\/[^\n]*/g, "");
}

function statisticsIn(code: string, file = "") {
  let scanned = withoutText(code);
  for (const allowed of ALLOWED[file] ?? []) scanned = scanned.replace(allowed, "");
  return FORBIDDEN.filter(([, pattern]) => pattern.test(scanned)).map(([name]) => name);
}

const SUM = "sum, the start of a mean";
const ORDER = "numeric sort, the start of an order statistic";
const POWER = "power, exponential, logarithm or root";

describe("no statistics in the chart components, the workspace or the domain", () => {
  it.each(sources)("%s computes no statistic", (name) => {
    expect(statisticsIn(readFileSync(root + name, "utf8"), name)).toEqual([]);
  });

  it.each(Object.keys(ALLOWED))("%s still needs each of its allowances", (name) => {
    const code = withoutText(readFileSync(root + name, "utf8"));
    for (const allowed of ALLOWED[name]!) expect(code).toMatch(new RegExp(allowed.source));
  });

  it.each([
    ["const q = values.sort((a, b) => a - b)[Math.floor(values.length / 2)];", [ORDER]],
    ["const q = values.toSorted((a: number, b: number) => a - b);", [ORDER]],
    ["rows.sort((x, y) => x.value - y.value);", [ORDER]],
    ["const bin = Math.floor((value - minimum) / width);", ["binning"]],
    ["const h = 1.06 * Math.sqrt(variance) * n ** -0.2;", ["spread", POWER]],
    ["const squares = (value - mean) ** 2;", [POWER]],
    ["const q1 = quantile(values, 0.25);", ["quantile or percentile"]],
    ["const total = rows.reduce((sum, row) => row.value + sum, 0);", [SUM]],
    ["const total = values.reduce((a: number, b: number) => a + b, 0);", [SUM]],
    ["const total = values.reduce((a, b) => { return a + b; }, 0);", [SUM]],
    ["let total = 0; for (const value of values) total += value;", [SUM]],
    ["let total = 0; for (const value of values) { total += value; }", [SUM]],
    ["t('say \"hi'); const sd = Math.sqrt(v); t('bye\"');", [POWER]],
    // Reading a value the plan carries is fine, and so are a chart type's name and words for people.
    ["`h = ${shown(mark.bandwidth)} (${mark.bandwidthRule})`", []],
    ['if (chart.type === "histogram") return;', []],
    ["const note = <p>Gaussian density</p>; // a histogram", []],
  ])("scans %s", (code, found) => {
    expect(statisticsIn(code)).toEqual(found);
  });

  it.each(sources)("%s would fail with a statistic in it", (name) => {
    const code = readFileSync(root + name, "utf8");
    for (const statistic of [
      "const sd = Math.sqrt(squares / (values.length - 1));",
      "const ordered = values.sort((a, b) => a - b);",
      "const mean = values.reduce((sum, value) => sum + value, 0) / values.length;",
    ]) expect(statisticsIn(`${code}\n${statistic}`, name)).toHaveLength(1);
  });
});
