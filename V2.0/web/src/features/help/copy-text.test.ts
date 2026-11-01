// @vitest-environment jsdom
import {afterEach, expect, it, vi} from "vitest";
import {copyText} from "./copy-text";
afterEach(()=>{document.body.replaceChildren();vi.restoreAllMocks();});
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
