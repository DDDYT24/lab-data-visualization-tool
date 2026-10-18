import { describe, expect, it } from "vitest";
import { scientificChartSpecSchema } from "./research-charts";

describe("scientific chart shared contract", () => {
  it.each(["grouped_strip", "paired", "violin"])("accepts %s with explicit stable parameters", (kind) => {
    expect(scientificChartSpecSchema.safeParse({
      schemaVersion: 1, kind, title: "Example", locale: "en", jitterSeed: "fixed", jitterWidth: 0.3,
      showPoints: true, summary: "none", summaryUnit: "experimental_unit",
      pairTimes: kind === "paired" ? ["before", "after"] : [], bandwidth: null, densityScale: "equal_width",
    }).success).toBe(true);
  });
  it.each([{ kind: "execute" }, { jitterWidth: 1 }, { bandwidth: 0 }, { formula: "x" },
    { pairTimes: ["a", "b", "c"] }, { locale: "unrecognized" }])("rejects invalid options %j", (body) => {
    expect(scientificChartSpecSchema.safeParse(body).success).toBe(false);
  });
});
