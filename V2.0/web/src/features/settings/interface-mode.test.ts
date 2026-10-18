// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { interfaceModeEvent, interfaceModeKey, loadInterfaceMode, saveInterfaceMode } from "./interface-mode";
import { defaultUserPreferences, loadUserPreferences, saveUserPreferences } from "./user-preferences";

describe("interface mode compatibility", () => {
  beforeEach(() => window.localStorage.clear());
  afterEach(() => vi.restoreAllMocks());

  it("keeps classic for existing users and invalid stored values", () => {
    expect(loadInterfaceMode()).toBe("classic");
    window.localStorage.setItem(interfaceModeKey, "future-layout");
    expect(loadInterfaceMode()).toBe("classic");
  });

  it("switches layout without modifying publication or appearance defaults", () => {
    saveUserPreferences({ ...defaultUserPreferences, dpi: 600, appearance: "dark", fontFamily: "Times New Roman" });
    const original = loadUserPreferences();
    const listener = vi.fn();
    window.addEventListener(interfaceModeEvent, listener);
    expect(saveInterfaceMode("overview")).toBe(true);
    expect(loadInterfaceMode()).toBe("overview");
    expect(listener).toHaveBeenCalledOnce();
    expect(loadUserPreferences()).toEqual(original);
    window.removeEventListener(interfaceModeEvent, listener);
  });

  it("survives disabled browser storage and reports non-persistence", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new DOMException("Blocked", "SecurityError"); });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new DOMException("Full", "QuotaExceededError"); });
    expect(loadInterfaceMode()).toBe("classic");
    expect(saveInterfaceMode("overview")).toBe(false);
  });
});
