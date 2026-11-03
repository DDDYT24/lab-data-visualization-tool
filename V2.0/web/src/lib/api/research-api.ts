import {
  preparationPlanSchema, researchPageSchema, researchPreviewSchema, researchSourcesSchema, researchViewSchema,
  type PreparationPlan,
} from "@/domain/research-preparation";
import { request } from "./labviz-api";

const path = (id: string) => `/projects/${encodeURIComponent(id)}/research`;

export const researchApi = {
  page(id: string, kind: "observations" | "samples", offset: number, revision: string | null, token?: string) {
    const query = new URLSearchParams({ kind, offset: String(offset), limit: kind === "samples" ? "200" : "100" });
    if (token) query.set("token", token);
    if (revision) query.set("revision", revision);
    return request(path(id) + "/page?" + query.toString(), researchPageSchema);
  },
  get(id: string) { return request(path(id), researchViewSchema); },
  inspect(id: string, files: File[]) {
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    return request(path(id) + "/inspect", researchSourcesSchema, { method: "POST", body });
  },
  preview(id: string, plan: PreparationPlan, files: File[], revision: string | null, signal?: AbortSignal) {
    preparationPlanSchema.parse(plan);
    const body = new FormData();
    body.append("plan", JSON.stringify(plan));
    if (revision) body.append("expectedRevision", revision);
    files.forEach((file) => body.append("files", file));
    return request(path(id) + "/preview", researchPreviewSchema, { method: "POST", body, signal });
  },
  apply(id: string, token: string, revision: string | null) {
    return request(path(id) + "/apply", researchViewSchema, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token, expectedRevision: revision }),
    });
  },
  history(id: string, direction: "undo" | "redo", revision: string) {
    return request(path(id) + "/history", researchViewSchema, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ direction, expectedRevision: revision }),
    });
  },
};
