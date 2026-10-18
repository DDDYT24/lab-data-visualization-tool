import { describe, expect, it } from "vitest";
import type { DataPreview } from "./api-contract";
import { defaultChartSpec, type ChartSpec } from "./chart-spec";
import { repairSingleValueChart } from "./single-value-chart";

const column = (field: string, kind: "number" | "text"): DataPreview["columns"][number] => ({
  field, kind, label: field, unit: null, nullable: false,
});
const oldChart: ChartSpec = {
  ...defaultChartSpec, title: "Saved voltage measurement",
  xAxis: { field: "voltage", title: "Voltage", unit: "mV" },
  yAxis: { field: "voltage", title: "Voltage", unit: "mV" },
  series: [{ ...defaultChartSpec.series[0], field: "voltage", label: "Measured voltage", color: "#DC2626" }],
  fitting: { ...defaultChartSpec.fitting, model: "linear", confidenceBand: true },
  uncertainty: { ...defaultChartSpec.uncertainty, mode: "standard-error" },
};

describe("saved single-variable chart compatibility", () => {
  it("reopens the old self-versus-self default as a distribution without losing publication settings", () => {
    const repaired = repairSingleValueChart(oldChart, [column("voltage", "number")]);
    expect(repaired.type).toBe("histogram");
    expect(repaired.title).toBe("Saved voltage measurement");
    expect(repaired.yAxis).toEqual({ field: "voltage", title: "Voltage", unit: "mV" });
    expect(repaired.series).toEqual(oldChart.series);
    expect(repaired.export).toEqual(oldChart.export);
    expect(repaired.fitting).toMatchObject({ model: "none", confidenceBand: false });
    expect(repaired.uncertainty.mode).toBe("none");
    expect(oldChart.type).toBe("line");
  });
  it("preserves a category-versus-value chart with one numeric variable", () => {
    const categories = { ...oldChart, type: "bar" as const, xAxis: { field: "sample", title: "Sample", unit: "" } };
    expect(repairSingleValueChart(categories, [column("sample", "text"), column("voltage", "number")])).toBe(categories);
  });
  it("preserves a genuine two-variable relationship", () => {
    const relationship = { ...oldChart, xAxis: { field: "time", title: "Time", unit: "s" } };
    expect(repairSingleValueChart(relationship, [column("time", "number"), column("voltage", "number")])).toBe(relationship);
  });
  it("preserves an explicitly saved distribution", () => {
    const distribution = { ...oldChart, type: "histogram" as const };
    expect(repairSingleValueChart(distribution, [column("voltage", "number")])).toBe(distribution);
  });
});
