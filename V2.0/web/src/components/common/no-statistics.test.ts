import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

// Charts here draw the API's RenderPlan. A statistic computed in the browser would be a second
// answer that the Matplotlib export cannot reproduce, so this folder must not compute one.
const FORBIDDEN: [string, RegExp][] = [
  ["quantile or percentile", /quantile|percentile/i],
  ["density estimate", /\bkde\b|kernel|gaussian|\bdensity\b|(?<!\.)\bbandwidth\b/i],
  ["spread", /variance|stdev|stddev|standardDeviation/i],
  ["binning", /histogram|binWidth|binCount|Math\.floor\(\s*\(\s*[\w.]+\s*-/i],
  ["numeric sort, the start of an order statistic", /\.sort\(\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*=>\s*\1\s*-\s*\2\s*\)/],
  ["exponential, logarithm or root", /Math\.(?:exp|log\d*|sqrt|pow)\b/],
];

const folder = fileURLToPath(new URL(".", import.meta.url));
const sources = readdirSync(folder).filter((name) => /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name));

function statisticsIn(code: string) {
  return FORBIDDEN.filter(([, pattern]) => pattern.test(code)).map(([name]) => name);
}

describe("no statistics in the chart components", () => {
  it.each(sources)("%s computes no statistic", (name) => {
    expect(statisticsIn(readFileSync(folder + name, "utf8"))).toEqual([]);
  });

  it("catches the code it is meant to keep out", () => {
    expect(statisticsIn("const q = values.sort((a, b) => a - b)[Math.floor(values.length / 2)];"))
      .toEqual(["numeric sort, the start of an order statistic"]);
    expect(statisticsIn("const bin = Math.floor((value - minimum) / width);")).toEqual(["binning"]);
    expect(statisticsIn("const h = 1.06 * Math.sqrt(variance) * n ** -0.2;"))
      .toEqual(["spread", "exponential, logarithm or root"]);
    expect(statisticsIn("const q1 = quantile(values, 0.25);")).toEqual(["quantile or percentile"]);
    // Reading a value the plan carries is fine.
    expect(statisticsIn("`h = ${shown(mark.bandwidth)} (${mark.bandwidthRule})`")).toEqual([]);
  });
});
