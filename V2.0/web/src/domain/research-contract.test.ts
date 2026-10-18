import { describe, expect, it } from "vitest";
import invalid from "../../../contracts/fixtures/v23/research.invalid.json";
import valid from "../../../contracts/fixtures/v23/research.valid.json";
import unknown from "../../../contracts/fixtures/v23/research.unknown-identity.json";
import { researchContractSchema } from "./research-contract";

describe("Research v1 prospective sidecar", () => {
  it("accepts the mixed four-panel researcher fixture", () => {
    expect(researchContractSchema.safeParse(valid).success).toBe(true);
  });
  it("preserves unknown legacy sample identity", () => {
    expect(researchContractSchema.safeParse(unknown).success).toBe(true);
    expect(unknown.analyses[0].independentUnitCount).toBeNull();
  });
  it.each(invalid)("rejects $case", ({ value }) => {
    expect(researchContractSchema.safeParse(value).success).toBe(false);
  });
});
