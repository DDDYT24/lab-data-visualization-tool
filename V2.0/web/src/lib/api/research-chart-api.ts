import { chartHistorySchema, scientificChartSchema, scientificChartSpecSchema, type ScientificChartSpec } from "@/domain/research-charts";
import { request } from "./labviz-api";

const path = (projectId: string) => `/projects/${encodeURIComponent(projectId)}/research/charts`;
export const researchChartApi = {
  list(projectId: string) { return request(path(projectId), chartHistorySchema); },
  get(projectId: string, id: string, offset = 0) {
    return request(path(projectId) + `/${encodeURIComponent(id)}?offset=${offset}`, scientificChartSchema);
  },
  create(projectId: string, revisionId: string, spec: ScientificChartSpec) {
    scientificChartSpecSchema.parse(spec);
    return request(path(projectId), scientificChartSchema, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ revisionId, spec }),
    });
  },
  artifact(projectId: string, id: string, format: "svg" | "png" | "pdf" | "json", download = false, preview = false, previewWidth = 960) {
    const base = (process.env.NEXT_PUBLIC_LABVIZ_API_URL ?? "/api/v1").replace(/\/$/, "");
    return base + path(projectId) + `/${encodeURIComponent(id)}/artifact/${format}?download=${download}&preview=${preview}&previewWidth=${previewWidth}`;
  },
};
