import { describe, expect, it } from "vitest";

import invalidFixtures from "../../../contracts/fixtures/project-spec-v1.invalid.json";
import validFixture from "../../../contracts/fixtures/project-spec-v1.valid.json";
import { projectSpecSchema } from "./project-spec";

describe("ProjectSpec v1 generated Zod contract", () => {
  it("accepts the shared valid fixture", () => {
    expect(projectSpecSchema.safeParse(validFixture).success).toBe(true);
  });

  it.each(invalidFixtures)("rejects $case", ({ value }) => {
    expect(projectSpecSchema.safeParse(value).success).toBe(false);
  });
});
