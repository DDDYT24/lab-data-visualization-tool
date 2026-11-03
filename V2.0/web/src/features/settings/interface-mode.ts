export type InterfaceMode = "classic" | "overview";
export const interfaceModeKey = "labviz:v23:interface-mode:v1";
export const interfaceModeEvent = "labviz:interface-mode-changed";

export function parseInterfaceMode(value: unknown): InterfaceMode | null {
  return value === "classic" || value === "overview" ? value : null;
}

export function loadInterfaceMode(): InterfaceMode {
  if (typeof window === "undefined") return "classic";
  try { return parseInterfaceMode(window.localStorage.getItem(interfaceModeKey)) ?? "classic"; }
  catch { return "classic"; }
}

export function saveInterfaceMode(mode: InterfaceMode): boolean {
  try {
    window.localStorage.setItem(interfaceModeKey, mode);
    window.dispatchEvent(new Event(interfaceModeEvent));
    return true;
  } catch { return false; }
}
