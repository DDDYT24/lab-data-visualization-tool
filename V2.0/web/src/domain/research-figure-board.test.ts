import { describe, expect, it } from "vitest";
import { figureBoardSpecSchema } from "./research-figure-board";

describe("figure layout contract", () => {
  it.each([1, 2, 4])("accepts %i independent saved-chart bindings", (count) => {
    expect(figureBoardSpecSchema.safeParse({
      schemaVersion: 1, title: "Figure", axisMode: "independent", panelChartIds: Array(count).fill("chart1"),
    }).success).toBe(true);
  });
  it.each(["shared_x", "shared_y", "shared_xy"])("accepts compatible-axis request %s", (axisMode) => {
    expect(figureBoardSpecSchema.safeParse({ panelChartIds: ["a", "b"], axisMode, palette: "grayscale", fontSize: 12, gridVisible: false }).success).toBe(true);
  });
  it.each([
    { panelChartIds: ["a"], fontSize: 100 }, { panelChartIds: ["a"], palette: "script" },
    { panelChartIds: [] }, { panelChartIds: ["a", "b", "c"] }, { panelChartIds: Array(5).fill("a") },
    { panelChartIds: [""] }, { panelChartIds: ["a"], axisMode: "shared" },
    { panelChartIds: ["a"], script: "execute" },
  ])("rejects unsupported layout %j", (body) => {
    expect(figureBoardSpecSchema.safeParse(body).success).toBe(false);
  });
});
