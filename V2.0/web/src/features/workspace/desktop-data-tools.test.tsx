// @vitest-environment jsdom
import React, {act} from "react";
import {createRoot, type Root} from "react-dom/client";
import {NextIntlClientProvider} from "next-intl";
import {afterEach, expect, it, vi} from "vitest";
import {DesktopDataTools} from "./desktop-data-tools";

(globalThis as typeof globalThis & {IS_REACT_ACT_ENVIRONMENT: boolean}).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root | undefined, container: HTMLDivElement;
const first = "a".repeat(32), second = "b".repeat(32);
let next: string | null;
async function mount({waiting=false, fail=false}={}) {
  next = second;
  Object.defineProperty(window, "labvizDesktopTransport", {configurable:true, value:{url:"http://127.0.0.1:51234/preferences",token:"test-only"}});
  vi.stubGlobal("fetch", vi.fn(async (url: URL, options: RequestInit) => {
    if (String(url).endsWith("/maintenance/select-copy")) {
      if (fail) return {ok:false,json:async()=>({message:"The restored copy has not completed verification"})};
      next = JSON.parse(String(options.body)).copyId;
      return {ok:true,json:async()=>({selected:true,nextCopyId:next})};
    }
    return {ok:true,json:async()=>({operation:{state:waiting?"waiting":"completed",action:"verify"},backups:[],profiles:{activeCopyId:null,nextCopyId:next,copies:[{copyId:first,profileId:"p1"},{copyId:second,profileId:"p2"}],unavailableCopies:1}})};
  }));
  container = document.createElement("div"); document.body.append(container); root = createRoot(container);
  await act(async()=>root!.render(<NextIntlClientProvider locale="en" messages={{}} timeZone="UTC"><DesktopDataTools/></NextIntlClientProvider>));
}
function button(text:string) {return Array.from(container.querySelectorAll("button")).find(b=>b.textContent===text)!;}
afterEach(async()=>{if(root) await act(async()=>root!.unmount());root=undefined;container?.remove();vi.unstubAllGlobals();});
it("shows current and next profiles and keeps older copies selectable after verification", async()=>{
  await mount();
  expect(container.querySelector('[data-testid="desktop-active-profile"]')?.textContent).toContain("Original profile");
  expect(container.querySelector('[data-testid="desktop-next-profile"]')?.textContent).toContain("bbbbbbbb");
  expect(container.textContent).toContain("Their files are retained");
  const select=Array.from(container.querySelectorAll('[role="combobox"]')).at(-1)!;
  await act(async()=>select.dispatchEvent(new MouseEvent("mousedown",{bubbles:true})));
  const option=Array.from(document.querySelectorAll('[role="option"]')).find(o=>o.textContent?.includes("aaaaaaaa"))!;
  await act(async()=>option.dispatchEvent(new MouseEvent("click",{bubbles:true})));
  await act(async()=>button("Use selected copy on next launch").click());
  expect(next).toBe(first);
  expect(container.querySelector('[data-testid="desktop-next-profile"]')?.textContent).toContain("aaaaaaaa");
  expect(container.querySelector('[data-testid="desktop-active-profile"]')?.textContent).toContain("Original profile");
  await act(async()=>button("Use original profile on next launch").click());expect(next).toBeNull();
  expect(container.textContent).toContain("Exit LabViz and reopen it");
});
it("disables profile switching while maintenance is waiting",async()=>{
  await mount({waiting:true}); expect(button("Use original profile on next launch").disabled).toBe(true);
  expect(button("Use selected copy on next launch").disabled).toBe(true);
});
it("reports a rejected selection without claiming success or changing the next profile",async()=>{
  await mount({fail:true});await act(async()=>button("Use original profile on next launch").click());
  expect(container.textContent).toContain("has not completed verification");
  expect(container.textContent).not.toContain("Exit LabViz and reopen it");expect(next).toBe(second);
});
