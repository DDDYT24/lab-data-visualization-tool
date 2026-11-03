import { z } from "zod";
import { request } from "./labviz-api";

const path = (id: string) => `/projects/${encodeURIComponent(id)}/research`;
const json = (value: unknown) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value) });
export const templateSchema = z.looseObject({ name: z.string(), panels: z.array(z.looseObject({ sourceColumns: z.array(z.string()) })) });
const templateRecord = z.object({ id: z.string(), spec: templateSchema });
const report = z.object({ panel: z.string(), status: z.enum(["ready", "blocked"]), observations: z.number(), unit: z.string(), message: z.string(), styles: z.unknown() });
const replacementSchema = z.object({ token: z.string(), expectedRevision: z.string().nullable(), panels: z.array(report), readyCount: z.number(), figureReady: z.boolean(), figureError: z.string().nullable(), notice: z.string() });
export const notesSchema = z.object({ legendEn: z.string(), legendZh: z.string(), methodsEn: z.string(), methodsZh: z.string() });
const manifestSchema = z.looseObject({ boardId: z.string(), originalUploadsIncluded: z.boolean(), identifiersIncluded: z.array(z.string()), panels: z.array(z.looseObject({ label: z.string(), revisionId: z.string(), dataRole: z.string(), sourceCsvRows: z.number() })) });
const outputPreviewSchema = z.object({ token: z.string(), manifest: manifestSchema, notes: notesSchema, files: z.array(z.object({ name: z.string(), bytes: z.number().nullable(), sha256: z.string().nullable() })) });
const outputSchema = z.object({ id: z.string(), boardId: z.string(), sha256: z.string(), createdAt: z.string() });
export type TemplateRecord = z.infer<typeof templateRecord>;
export type ReplacementPreview = z.infer<typeof replacementSchema>;
export type OutputPreview = z.infer<typeof outputPreviewSchema>;
export type OutputNotes = z.infer<typeof notesSchema>;
export type SavedOutput = z.infer<typeof outputSchema>;
export const researchReuseApi = {
  templates(id: string) { return request(path(id) + "/templates", z.object({ templates: z.array(templateRecord) })); },
  create(id: string, boardId: string, name: string) { return request(path(id) + "/templates", templateRecord, json({ boardId, name })); },
  import(id: string, spec: unknown) { return request(path(id) + "/templates/import", templateRecord, json(spec)); },
  templateUrl(id: string, template: string) { return `/api/v1${path(id)}/templates/${encodeURIComponent(template)}`; },
  preview(id: string, template: string, file: File, mapping: Record<string, string>, revision: string | null) {
    const body = new FormData(); body.append("file", file); body.append("mapping", JSON.stringify(mapping));
    if (revision) body.append("expectedRevision", revision);
    return request(path(id) + `/templates/${encodeURIComponent(template)}/preview`, replacementSchema, { method: "POST", body });
  },
  apply(id: string, token: string, expectedRevision: string | null) { return request(path(id) + "/template-apply", z.object({ boardId: z.string().nullable(), revisionId: z.string().nullable(), chartIds: z.array(z.string()) }), json({ token, expectedRevision })); },
  previewOutput(id: string, boardId: string) { return request(path(id) + "/outputs/preview", outputPreviewSchema, json({ boardId })); },
  freeze(id: string, token: string, notes: OutputNotes) { return request(path(id) + "/outputs", outputSchema, json({ token, notes, reviewedIncludedData: true })); },
  outputs(id: string) { return request(path(id) + "/outputs", z.object({ outputs: z.array(outputSchema) })); },
  download(id: string, output: string) { return `/api/v1${path(id)}/outputs/${encodeURIComponent(output)}/download`; },
};
