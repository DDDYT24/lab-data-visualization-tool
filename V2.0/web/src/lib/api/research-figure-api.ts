import { z } from "zod";
import {
  figureInteractionSchema, figureSourceSchema, figureBoardSchema, figureBoardHistorySchema, figureBoardSpecSchema, type FigureBoardSpec,
} from "@/domain/research-figure-board";
import { request } from "./labviz-api";

const path = (id: string) => `/projects/${encodeURIComponent(id)}/research/figure-boards`;
export const researchFigureApi = {
  sources(id: string) { return request(`/projects/${encodeURIComponent(id)}/research/figure-sources`, figureSourceSchema); },
  capture(id: string, panel: number, sampleField: string) { return request(`/projects/${encodeURIComponent(id)}/research/figure-sources?panel=${panel}${sampleField ? `&sampleField=${encodeURIComponent(sampleField)}` : ""}`, z.object({ id: z.string() }), { method: "POST" }); },
  artifact(id: string, board: string, format: string, download = false, preview = false) { return `/api/v1${path(id)}/${encodeURIComponent(board)}/artifact/${format}?download=${download}&preview=${preview}`; },
  interaction(id: string, board: string, params: URLSearchParams, signal?: AbortSignal) { return request(`${path(id)}/${encodeURIComponent(board)}/interaction?${params}`, figureInteractionSchema, { signal }); },
  list(id: string) { return request(path(id), figureBoardHistorySchema); },
  get(id: string, boardId: string) {
    return request(path(id) + "/" + encodeURIComponent(boardId), figureBoardSchema);
  },
  retry(id: string, boardId: string) {
    return request(`${path(id)}/${encodeURIComponent(boardId)}/retry`, figureBoardSchema, { method: "POST" });
  },
  save(id: string, spec: FigureBoardSpec) {
    figureBoardSpecSchema.parse(spec);
    return request(path(id) + "?deferArtifacts=true", figureBoardSchema, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(spec),
    });
  },
};
