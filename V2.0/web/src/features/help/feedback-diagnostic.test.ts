import { describe, expect, it } from "vitest";

import { buildFeedbackDiagnostic, serializeFeedbackDiagnostic } from "./feedback-diagnostic";

describe("privacy-safe feedback diagnostics", () => {
  it("keeps only the approved summary fields and generalizes identifiers", () => {
    const diagnostic = buildFeedbackDiagnostic({
      category: "bug",
      description: "The chart preview is blank.",
      pathname: "/workspace/private-project-123",
      userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/140.0.0.0",
    });

    expect(diagnostic).toEqual({
      appVersion: "2.2.0",
      category: "bug",
      platform: "Windows",
      browser: "Chrome",
      screen: "workspace",
      description: "The chart preview is blank.",
    });
    expect(JSON.stringify(diagnostic)).not.toContain("private-project-123");
  });

  it("does not add source rows, filenames, email addresses, or persistent IDs", () => {
    const serialized = serializeFeedbackDiagnostic({
      category: "usability",
      description: "Please improve the empty state.",
      pathname: "/share/secret-token",
      userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) Safari/617.1",
    });

    expect(serialized).not.toMatch(/rowId|filename|email|projectId|secret-token|response_mV/);
    expect(serialized).toContain('"screen": "shared-chart"');
  });
});
