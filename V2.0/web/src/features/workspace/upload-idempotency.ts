export type UploadIdempotencyState = {
  key: string;
  signature: string;
};

export function uploadRequestSignature(
  file: File,
  options: {
    sheetName?: string | null;
    headerRow?: number | null;
    experimentTitle?: string | null;
    runLabel?: string | null;
    replicateId?: string | null;
    batchId?: string | null;
    experimentRunId?: string | null;
  } = {},
): string {
  return JSON.stringify([
    file.name,
    file.size,
    file.lastModified,
    file.type,
    options.sheetName ?? null,
    options.headerRow ?? null,
    options.experimentTitle ?? null,
    options.runLabel ?? null,
    options.replicateId ?? null,
    options.batchId ?? null,
    options.experimentRunId ?? null,
  ]);
}

export function getUploadIdempotencyState(
  current: UploadIdempotencyState | null,
  signature: string,
): UploadIdempotencyState {
  if (current?.signature === signature) return current;
  return { signature, key: crypto.randomUUID() };
}
