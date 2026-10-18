export type ResearchTask = "data" | "quality" | "analysis" | "figures" | "outputs";
export const workspaceModeEvent = "labviz:workspace-mode";

export function requestWorkspaceMode(research: boolean) {
  window.dispatchEvent(new CustomEvent<boolean>(workspaceModeEvent, { detail: research }));
}

export function researchTaskFromQuery(value: unknown): ResearchTask {
  return value === "quality" || value === "analysis" || value === "figures" || value === "outputs"
    ? value : "data";
}

// Only navigation is remembered; data, drafts and frozen results stay in their existing stores.
export function rememberResearchNavigation(research: boolean, task?: ResearchTask) {
  const url = new URL(window.location.href);
  if (research) url.searchParams.set("workspace", "research");
  else url.searchParams.delete("workspace");
  if (task) url.searchParams.set("researchTask", task);
  window.history.replaceState(window.history.state, "", url);
}
