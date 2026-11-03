/**
 * Prospective sidecar validation only. Existing ProjectSpec v1 remains authoritative
 * for currently persisted projects; this module does not migrate or save them.
 */
import { z } from "zod";
import jsonSchema from "../../../contracts/research-v1.schema.json";

const structuralSchema = z.fromJSONSchema(
  jsonSchema as Parameters<typeof z.fromJSONSchema>[0],
);

// JSON Schema carries field constraints; graph invariants are checked in both runtimes.
// Their shared positive/negative corpus must pass whenever either validator changes.
export const researchContractSchema = structuralSchema.superRefine((input, ctx) => {
  const value = input as ResearchGraph;
  const fail = (message: string) => ctx.addIssue({ code: "custom", message });
  const unique = (ids: string[]) => ids.length === new Set(ids).size;
  const collections = [value.sources, value.samples, value.acquisitions,
    value.observations, value.processing, value.analyses, value.figure.panels];
  if (collections.some((items) => !unique(items.map((item) => item.id)))) fail("duplicate_id");
  const sources = new Set(value.sources.map((item) => item.id));
  const samples = new Map(value.samples.map((item) => [item.id, item]));
  const acquisitions = new Set(value.acquisitions.map((item) => item.id));
  const observations = new Map(value.observations.map((item) => [item.id, item]));
  const analyses = new Set(value.analyses.map((item) => item.id));
  for (const sample of value.samples) {
    const seen = new Set([sample.id]);
    let parent = sample.parentSampleId;
    while (parent !== null) {
      if (!samples.has(parent) || seen.has(parent)) { fail("invalid_sample_parent"); break; }
      seen.add(parent);
      parent = samples.get(parent)!.parentSampleId;
    }
  }
  for (const observation of value.observations) {
    if (observation.sampleId !== null && !samples.has(observation.sampleId)) fail("unknown_sample");
    if (!acquisitions.has(observation.acquisitionId)) fail("unknown_acquisition");
    if (observation.lineage.some((ref) => !sources.has(ref.sourceId))) fail("unknown_source");
    if (observation.state === "observed" && observation.value === null) fail("observed_requires_value");
    if (["missing", "not_measured", "failed"].includes(observation.state) && observation.value !== null)
      fail("absent_state_requires_null");
    if (["below_detection", "below_quantification"].includes(observation.state) && observation.limit === null)
      fail("qualified_state_requires_limit");
  }
  for (const analysis of value.analyses) {
    const included = analysis.inputObservationIds;
    const excluded = analysis.excluded.map((item) => item.observationId);
    if (analysis.dataRevision !== value.dataRevision || !unique(included) || !unique(excluded)
      || included.some((id) => excluded.includes(id))
      || [...included, ...excluded].some((id) => !observations.has(id))) {
      fail("invalid_analysis_inputs"); continue;
    }
    const selected = included.map((id) => observations.get(id)!);
    if (selected.some((item) => item.state !== "observed")) fail("unsupported_measurement_state");
    const sampleIds = new Set(selected.flatMap((item) => item.sampleId === null ? [] : [item.sampleId]));
    const units = new Set([...sampleIds].map((id) => samples.get(id)?.experimentalUnitId ?? null));
    const unknown = sampleIds.size === 0 || selected.some((item) => item.sampleId === null) || units.has(null);
    if (analysis.sampleCount !== sampleIds.size || analysis.observationCount !== selected.length
      || analysis.independentUnitCount !== (unknown ? null : units.size)) fail("incorrect_analysis_counts");
    if (analysis.analysisUnit === "experimental_unit" && unknown) fail("unknown_experimental_unit");
  }
  const panels = value.figure.panels;
  if (panels.length === 3 || panels.some((panel, index) => panel.id !== "ABCD"[index]))
    fail("invalid_panel_layout");
  for (const panel of panels) {
    if (panel.dataRevision !== value.dataRevision) fail("unknown_panel_revision");
    if (panel.analysisId !== null && !analyses.has(panel.analysisId)) fail("unknown_analysis");
  }
  for (const link of value.figure.axisLinks) {
    if (!unique(link.panelIds)) fail("duplicate_axis_panel");
    const linked = link.panelIds.map((id) => panels.find((panel) => panel.id === id));
    if (linked.some((panel) => !panel)) { fail("unknown_axis_panel"); continue; }
    const units = new Set(linked.map((panel) => link.axis === "x" ? panel!.xUnit : panel!.yUnit));
    if (units.size !== 1) fail("incompatible_axis_units");
  }
});

// A narrow projection for graph validation; JSON Schema validates every field first.
type Identified = { id: string };
type ResearchGraph = {
  dataRevision: string;
  sources: Identified[];
  samples: (Identified & { parentSampleId: string | null; experimentalUnitId: string | null })[];
  acquisitions: Identified[];
  observations: (Identified & { sampleId: string | null; acquisitionId: string;
    state: string; value: number | null; limit: number | null; lineage: { sourceId: string }[] })[];
  processing: Identified[];
  analyses: (Identified & { dataRevision: string; inputObservationIds: string[];
    excluded: { observationId: string }[]; sampleCount: number; observationCount: number;
    independentUnitCount: number | null; analysisUnit: string })[];
  figure: { panels: (Identified & { dataRevision: string; analysisId: string | null;
    xUnit: string; yUnit: string })[]; axisLinks: { axis: string; panelIds: string[] }[] };
};
