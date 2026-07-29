import { describe, expect, it } from "vitest";

import { chartSpecSchema, defaultChartSpec } from "./chart-spec";

describe("chartSpecSchema", () => {
  it("accepts the default serializable chart contract", () => {
    expect(chartSpecSchema.parse(defaultChartSpec)).toEqual(defaultChartSpec);
  });

  it("keeps the core V2.0 panel limit at four", () => {
    const result = chartSpecSchema.safeParse({
      ...defaultChartSpec,
      panelCount: 5,
    });
    expect(result.success).toBe(false);
  });

  it("rejects library-specific or malformed colors", () => {
    const result = chartSpecSchema.safeParse({
      ...defaultChartSpec,
      series: [{ ...defaultChartSpec.series[0], color: "blue" }],
    });
    expect(result.success).toBe(false);
  });

  it("keeps long-format grouping optional and serializable", () => {
    expect(
      chartSpecSchema.parse({ ...defaultChartSpec, groupField: "sample" }).groupField,
    ).toBe("sample");
    expect(chartSpecSchema.parse(defaultChartSpec).groupField).toBeNull();
  });
});
