import { describe, expect, it } from "vitest";

import {
  WEB_FILE_LIMIT_BYTES,
  formatBytes,
  validateWebFile,
} from "./file-policy";

describe("validateWebFile", () => {
  it("accepts supported extensions case-insensitively", () => {
    expect(validateWebFile({ name: "experiment.XLSX", size: 1024 })).toEqual({
      ok: true,
    });
  });

  it("accepts a file exactly at the website limit", () => {
    expect(
      validateWebFile({ name: "experiment.csv", size: WEB_FILE_LIMIT_BYTES }),
    ).toEqual({ ok: true });
  });

  it("accepts the reported 1.87 MB CSV regression case", () => {
    expect(
      validateWebFile({ name: "experiment.csv", size: 1_960_000 }),
    ).toEqual({ ok: true });
  });

  it("rejects files above the website limit before upload", () => {
    expect(
      validateWebFile({
        name: "experiment.csv",
        size: WEB_FILE_LIMIT_BYTES + 1,
      }),
    ).toEqual({ ok: false, reason: "too-large" });
  });

  it("rejects empty and unsupported files", () => {
    expect(validateWebFile({ name: "empty.csv", size: 0 })).toEqual({
      ok: false,
      reason: "empty",
    });
    expect(validateWebFile({ name: "notes.docx", size: 100 })).toEqual({
      ok: false,
      reason: "invalid-type",
    });
  });
});

describe("formatBytes", () => {
  it("uses readable binary units", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(2 * 1024 * 1024)).toBe("2.0 MB");
  });
});
