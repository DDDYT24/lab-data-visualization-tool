import { describe, expect, it } from "vitest";
import { preparationPlanSchema } from "./research-preparation";

describe("generated executable research preparation contract", () => {
  const base = { mapping: { value: "reading", sample: "sample" } };
  it("accepts bounded typed operations and explicit pair order", () => {
    expect(preparationPlanSchema.safeParse({ ...base, validatePairs: true, pairTimes: ["before", "after"],
      steps: [{ kind: "filter", field: "group", values: ["control"] },
        { kind: "convert_unit", valueField: "reading", unitField: "unit", toUnit: "V" }] }).success).toBe(true);
  });
  it.each([
    { ...base, steps: [{ kind: "execute", code: "anything" }] },
    { ...base, steps: [{ kind: "filter", field: "x", values: [] }] },
    { ...base, mapping: { value: "x", inferIndependentUnits: true } },
    { ...base, pairTimes: ["a", "b", "c"] },
    { ...base, exclusions: [{ observationId: "one", reason: "" }] },
  ])("rejects invalid executable plans", (value) => {
    expect(preparationPlanSchema.safeParse(value).success).toBe(false);
  });
});
