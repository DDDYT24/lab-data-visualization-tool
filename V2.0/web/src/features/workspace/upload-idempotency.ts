export type UploadIdempotencyState = {
  key: string;
  signature: string;
};

export function uploadRequestSignature(
  file: File,
  options: { sheetName?: string | null; headerRow?: number | null } = {},
): string {
  return JSON.stringify([
    file.name,
    file.size,
    file.lastModified,
    file.type,
    options.sheetName ?? null,
    options.headerRow ?? null,
  ]);
}

export function getUploadIdempotencyState(
  current: UploadIdempotencyState | null,
  signature: string,
): UploadIdempotencyState {
  if (current?.signature === signature) return current;
  return { signature, key: crypto.randomUUID() };
}
