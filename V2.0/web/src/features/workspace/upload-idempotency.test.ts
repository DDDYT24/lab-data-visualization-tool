import { afterEach, describe, expect, it, vi } from "vitest";

import {
  getUploadIdempotencyState,
  uploadRequestSignature,
} from "./upload-idempotency";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("upload idempotency state", () => {
  it("reuses a key for the same file and import options", () => {
    const file = new File(["a,b\n1,2"], "data.csv", {
      type: "text/csv",
      lastModified: 123,
    });
    const signature = uploadRequestSignature(file, {
      sheetName: null,
      headerRow: 1,
    });
    const first = getUploadIdempotencyState(null, signature);
    const replay = getUploadIdempotencyState(first, signature);

    expect(replay).toBe(first);
  });

  it("rotates a key when the import intent changes", () => {
    vi.stubGlobal("crypto", {
      randomUUID: vi.fn()
        .mockReturnValueOnce("key-1")
        .mockReturnValueOnce("key-2"),
    });
    const first = getUploadIdempotencyState(null, "first");
    const changed = getUploadIdempotencyState(first, "second");

    expect(first.key).toBe("key-1");
    expect(changed.key).toBe("key-2");
    expect(changed).not.toBe(first);
  });
});
