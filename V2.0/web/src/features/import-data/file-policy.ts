export const WEB_FILE_LIMIT_BYTES = 50 * 1024 * 1024;

const supportedExtensions = new Set(["xlsx", "csv", "tsv", "txt", "json"]);

export type FilePolicyResult =
  | { ok: true }
  | { ok: false; reason: "invalid-type" | "too-large" | "empty" };

type FileDescriptor = {
  name: string;
  size: number;
};

export function validateWebFile(file: FileDescriptor): FilePolicyResult {
  if (file.size === 0) {
    return { ok: false, reason: "empty" };
  }

  if (file.size > WEB_FILE_LIMIT_BYTES) {
    return { ok: false, reason: "too-large" };
  }

  const extension = file.name.split(".").pop()?.toLowerCase();
  if (!extension || !supportedExtensions.has(extension)) {
    return { ok: false, reason: "invalid-type" };
  }

  return { ok: true };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
