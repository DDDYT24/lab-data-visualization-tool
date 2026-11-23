import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

// Charts here draw the API's RenderPlan, and the workspace shows the API's results. A statistic
// computed in the browser would be a second answer that the Matplotlib export cannot reproduce.
const FORBIDDEN: [string, RegExp][] = [
  ["quantile or percentile", /quantile|percentile/i],
  ["density estimate", /\bkde\b|kernel|gaussian|\bdensity\b|(?<!\.)\bbandwidth\b/i],
  ["spread", /variance|stdev|stddev|standardDeviation/i],
  ["sum, the start of a mean", /\.reduce\(\s*\(\s*(\w+)\s*,[^)]*\)\s*=>\s*(?:\1\s*\+|\w[\w.]*\s*\+\s*\1\b)/],
  ["binning", /histogram|binWidth|binCount|Math\.floor\(\s*\(\s*[\w.]+\s*-/i],
  ["numeric sort, the start of an order statistic", /\.sort\(\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*=>\s*\1\s*-\s*\2\s*\)/],
  ["exponential, logarithm or root", /Math\.(?:exp|log\d*|sqrt|pow)\b/],
];

const root = fileURLToPath(new URL("../../", import.meta.url));
const sources = ["components/common", "features/workspace", "domain"].flatMap((folder) => readdirSync(root + folder)
  .filter((name) => /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name))
  .map((name) => `${folder}/${name}`));

// Names of things the API computes and drawing arithmetic, as they read once quoted text is gone.
const ALLOWED: Record<string, RegExp[]> = {
  // How many y axes the panels before this one have.
  "components/common/render-plan-option.ts": [/\.reduce\(\(sum, item\) => sum \+ \(item\.axes\.y2 \? 2 : 1\), 0\)/g],
  // Where a value sits along a log axis: the same rule as plan_axes.axis_position.
  "domain/render-plan.ts": [/Math\.log10\((?:value|1 \+ )/g, /bandwidth: number;/g, /(?:density|histogram|quantile): string \| null;/g],
  "domain/research-charts.ts": [/bandwidth: (?:number \| null;|z\.number\(\),)/g],
  "domain/api-contract.ts": [/histograms: z\.array\(/g],
  // The bandwidth a user types is a setting sent to the API; the density it made is described.
  "features/workspace/research-chart-workspace.tsx": [
    /\[bandwidth, setBandwidth\]/g, /\bsecond, bandwidth, scale\b/g, /value=\{bandwidth\}/g, / · Gaussian KDE · h=/g,
    /bandwidth: takes\(""\) && bandwidth !== "" \? Number\(bandwidth\) : null,/g, /`Density describes observations, /g,
  ],
};

function unquoted(code: string) {
  return code.replace(/"(?:[^"\\\n]|\\.)*"/g, '""');
}

function statisticsIn(code: string, file = "") {
  let scanned = unquoted(code);
  for (const allowed of ALLOWED[file] ?? []) scanned = scanned.replace(allowed, "");
  return FORBIDDEN.filter(([, pattern]) => pattern.test(scanned)).map(([name]) => name);
}

describe("no statistics in the chart components, the workspace or the domain", () => {
  it.each(sources)("%s computes no statistic", (name) => {
    expect(statisticsIn(readFileSync(root + name, "utf8"), name)).toEqual([]);
  });

  it.each(Object.keys(ALLOWED))("%s still needs each of its allowances", (name) => {
    const code = unquoted(readFileSync(root + name, "utf8"));
    for (const allowed of ALLOWED[name]!) expect(code).toMatch(new RegExp(allowed.source));
  });

  it("catches the code it is meant to keep out", () => {
    expect(statisticsIn("const q = values.sort((a, b) => a - b)[Math.floor(values.length / 2)];"))
      .toEqual(["numeric sort, the start of an order statistic"]);
    expect(statisticsIn("const bin = Math.floor((value - minimum) / width);")).toEqual(["binning"]);
    expect(statisticsIn("const h = 1.06 * Math.sqrt(variance) * n ** -0.2;"))
      .toEqual(["spread", "exponential, logarithm or root"]);
    expect(statisticsIn("const q1 = quantile(values, 0.25);")).toEqual(["quantile or percentile"]);
    expect(statisticsIn("const total = rows.reduce((sum, row) => row.value + sum, 0);")).toEqual(["sum, the start of a mean"]);
    // Reading a value the plan carries is fine, and so is naming a chart type.
    expect(statisticsIn("`h = ${shown(mark.bandwidth)} (${mark.bandwidthRule})`")).toEqual([]);
    expect(statisticsIn('if (chart.type === "histogram") return;')).toEqual([]);
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
