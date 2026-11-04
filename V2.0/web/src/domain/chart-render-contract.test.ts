import { describe, expect, it } from "vitest";

import { densePointThreshold, geometryContract } from "./chart-render-contract";

describe("geometry contract", () => {
  it("exposes the rule names research plans cite", () => {
    expect(geometryContract.contractVersion).toBe("chart-render-v2");
    expect(geometryContract.axis.rule).toBe("nice-1-2-5-v1");
    expect(geometryContract.jitter.rule).toBe("sha256-uniform-v1");
    expect(geometryContract.density.rule).toBe("gaussian-kde-v1");
    expect(densePointThreshold).toBe(geometryContract.points.denseThreshold);
  });
});
