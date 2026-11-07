// @vitest-environment jsdom
import {afterEach, expect, it, vi} from "vitest";
import {copyText} from "./copy-text";
afterEach(()=>{document.body.replaceChildren();vi.restoreAllMocks();vi.unstubAllGlobals();delete window.labvizDesktopTransport;});
function clipboard(writeText?: (text:string)=>Promise<void>) {
  Object.defineProperty(navigator,"clipboard",{configurable:true,value:writeText?{writeText}:undefined});
}
it("copies exact Chinese text with the modern API without using the system clipboard",async()=>{
  const write=vi.fn(async()=>{});clipboard(write);
  expect(await copyText("科研资料\nvalue,4.5")).toBe(true);expect(write).toHaveBeenCalledWith("科研资料\nvalue,4.5");
});
it("falls back after permission rejection and restores focus and input selection",async()=>{
  clipboard(async()=>{throw new Error("denied");});
  const input=document.createElement("input");input.value="retained draft";document.body.append(input);input.focus();input.setSelectionRange(2,7);
  Object.defineProperty(document,"execCommand",{configurable:true,value:vi.fn(()=>{expect(document.querySelector("textarea")?.value).toBe("exact text");return true;})});
  expect(await copyText("exact text")).toBe(true);expect(document.activeElement).toBe(input);
  expect([input.selectionStart,input.selectionEnd]).toEqual([2,7]);expect(document.querySelector("textarea")).toBeNull();
});
it.each([false,"throw"])("returns failure and removes the temporary field when fallback fails: %s",async result=>{
  clipboard();Object.defineProperty(document,"execCommand",{configurable:true,value:()=>{if(result==="throw") throw new Error("unavailable");return false;}});
  expect(await copyText("private draft")).toBe(false);expect(document.querySelector("textarea")).toBeNull();
});
it("requires the authenticated native bridge to verify an exact desktop copy",async()=>{
  const write=vi.fn(async()=>{});clipboard(write);
  Object.defineProperty(window,"labvizDesktopTransport",{configurable:true,value:{url:"http://127.0.0.1:51234/preferences",token:"test-only"}});
  const fetch=vi.fn(async()=>({ok:true,json:async()=>({copied:true,verified:true})}));vi.stubGlobal("fetch",fetch);
  const input=document.createElement("input");document.body.append(input);input.focus();
  expect(await copyText("科研资料\nvalue,4.5")).toBe(true);
  const [url,options]=fetch.mock.calls[0] as unknown as [URL,RequestInit];
  expect(String(url)).toBe("http://127.0.0.1:51234/clipboard/write");
  expect(options.headers).toEqual({"Content-Type":"application/json","X-LabViz-Desktop":"test-only"});
  expect(JSON.parse(options.body as string)).toEqual({text:"科研资料\nvalue,4.5"});
  expect(write).not.toHaveBeenCalled();expect(document.activeElement).toBe(input);
});
it.each(["busy","unverified","network"])("reports desktop copy failure without a browser fallback: %s",async mode=>{
  const write=vi.fn(async()=>{});clipboard(write);
  const fallback=vi.fn(()=>true);Object.defineProperty(document,"execCommand",{configurable:true,value:fallback});
  Object.defineProperty(window,"labvizDesktopTransport",{configurable:true,value:{url:"http://127.0.0.1:51234/preferences",token:"test-only"}});
  vi.stubGlobal("fetch",vi.fn(async()=>{if(mode==="network")throw Error("unavailable");return {ok:mode!=="busy",json:async()=>({copied:true,verified:false})};}));
  const input=document.createElement("input");input.value="retained draft";document.body.append(input);input.focus();input.setSelectionRange(2,7);
  expect(await copyText("synthetic")).toBe(false);expect(write).not.toHaveBeenCalled();expect(fallback).not.toHaveBeenCalled();
  expect(document.activeElement).toBe(input);expect([input.selectionStart,input.selectionEnd]).toEqual([2,7]);
});
