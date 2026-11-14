// @vitest-environment jsdom
import React, {act} from "react";
import {Blob as NodeBlob} from "node:buffer";
import {createRoot, type Root} from "react-dom/client";
import {NextIntlClientProvider} from "next-intl";
import {afterEach, expect, it, vi} from "vitest";
import {DesktopPdfPreview} from "./desktop-pdf-preview";
(globalThis as typeof globalThis & {IS_REACT_ACT_ENVIRONMENT:boolean}).IS_REACT_ACT_ENVIRONMENT=true;
let root:Root|undefined,container:HTMLDivElement;
const href="/api/v1/projects/p/research/charts/c/artifact/pdf?download=false";
const create=vi.fn(()=>"blob:http://localhost/validated-pdf"),revoke=vi.fn();
async function mount(url=href) {
  create.mockClear();revoke.mockClear();
  Object.defineProperty(window,"labvizDesktopTransport",{configurable:true,value:{url:"http://localhost/preferences",token:"test-only"}});
  Object.defineProperty(URL,"createObjectURL",{configurable:true,value:create});
  Object.defineProperty(URL,"revokeObjectURL",{configurable:true,value:revoke});
  container=document.createElement("div");document.body.append(container);root=createRoot(container);
  await act(async()=>root!.render(<NextIntlClientProvider locale="en" timeZone="UTC" messages={{}}><DesktopPdfPreview href={url}/></NextIntlClientProvider>));
  await act(async()=>container.querySelector("button")!.click());
}
function response(type="application/pdf",body="%PDF-1.7\nexact frozen bytes") {
  return {ok:true,headers:new Headers({"content-type":type}),blob:async()=>new NodeBlob([body],{type})};
}
afterEach(async()=>{if(root) await act(async()=>root!.unmount());root=undefined;container?.remove();vi.unstubAllGlobals();});
it("previews validated PDF bytes with a temporary URL while API frame protection remains untouched",async()=>{
  vi.stubGlobal("fetch",vi.fn(async()=>response()));await mount();
  expect(document.querySelector("iframe")?.getAttribute("src")).toBe("blob:http://localhost/validated-pdf");
  expect(create).toHaveBeenCalledOnce();
  const close=Array.from(document.querySelectorAll("button")).find(b=>b.textContent==="Return to workspace")!;
  await act(async()=>close.click());expect(revoke).toHaveBeenCalledWith("blob:http://localhost/validated-pdf");
});
it.each([["text/html","%PDF-fake"],["application/pdf","not a PDF"]])("rejects a bad media type or body: %s",async(type,body)=>{
  vi.stubGlobal("fetch",vi.fn(async()=>response(type,body)));await mount();
  expect(document.querySelector('[role="alert"]')?.textContent).toContain("Could not preview");
  expect(document.querySelector("iframe")).toBeNull();expect(create).not.toHaveBeenCalled();
});
it("rejects an external source before sending a request",async()=>{
  const fetch=vi.fn();vi.stubGlobal("fetch",fetch);await mount("https://example.invalid/artifact.pdf");
  expect(fetch).not.toHaveBeenCalled();expect(document.querySelector("iframe")).toBeNull();
});
it("aborts a closed preview and ignores a late response without leaking a document URL",async()=>{
  let resolve!:(value:ReturnType<typeof response>)=>void;
  const fetch=vi.fn(()=>new Promise(r=>{resolve=r;}));vi.stubGlobal("fetch",fetch);await mount();
  const signal=(fetch.mock.calls[0] as unknown as [URL,RequestInit])[1].signal!;
  const close=Array.from(document.querySelectorAll("button")).find(b=>b.textContent==="Return to workspace")!;
  await act(async()=>close.click());expect(signal.aborted).toBe(true);
  await act(async()=>resolve(response()));expect(create).not.toHaveBeenCalled();
});
it("closes from the native Escape signal after the PDF frame takes focus and restores the entry",async()=>{
  vi.stubGlobal("fetch",vi.fn(async()=>response()));await mount();
  const frame=document.querySelector("iframe")!;
  frame.focus();expect(document.activeElement).toBe(frame);
  await act(async()=>window.dispatchEvent(new Event("labviz:desktop-escape")));
  expect(document.querySelector("iframe")).toBeNull();
  expect(revoke).toHaveBeenCalledWith("blob:http://localhost/validated-pdf");
  await act(async()=>new Promise(resolve=>setTimeout(resolve,250)));
  expect(document.activeElement).toBe(container.querySelector("button"));
  const outside=document.createElement("button");document.body.append(outside);outside.focus();
  await act(async()=>window.dispatchEvent(new Event("labviz:desktop-escape")));
  expect(document.activeElement).toBe(outside);outside.remove();
});
it("aborts an in-flight source when the native Escape signal closes the preview",async()=>{
  const fetch=vi.fn(()=>new Promise(()=>{}));vi.stubGlobal("fetch",fetch);await mount();
  const signal=(fetch.mock.calls[0] as unknown as [URL,RequestInit])[1].signal!;
  await act(async()=>window.dispatchEvent(new Event("labviz:desktop-escape")));
  expect(signal.aborted).toBe(true);expect(create).not.toHaveBeenCalled();
});
