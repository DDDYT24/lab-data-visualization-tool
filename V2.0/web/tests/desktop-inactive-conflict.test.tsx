// @vitest-environment jsdom
import React, {act, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {NextIntlClientProvider} from 'next-intl';
import {expect, it, vi} from 'vitest';
import {DesktopSession, useDesktopDraft} from '../src/features/workspace/desktop-session';

(globalThis as typeof globalThis & {IS_REACT_ACT_ENVIRONMENT:boolean}).IS_REACT_ACT_ENVIRONMENT=true;

it('keeps a parked draft through another checkpoint, then warns when its changed revision is opened',async()=>{
  const profileId='inactive-conflict-test',key='chart:p1';
  const snapshot={format:'labviz-desktop-checkpoint-v1',profileId,route:'/workspace/p1',checkpointId:'old',entries:{[key]:{version:1,projectId:'p1',base:{research:'r1'},value:{title:'Old unsaved draft'}}}};
  Object.defineProperty(window,'labvizDesktop',{value:{profileId},configurable:true});
  Object.defineProperty(window,'labvizDesktopTransport',{value:{url:'http://127.0.0.1:52525/',token:'test-only',session:{initial:snapshot,warning:null}},configurable:true});
  window.labvizDesktopPending=()=>0;window.labvizDesktopFlushPreferences=async()=>{};
  let saved:{checkpointId:string;entries:Record<string,{value:{title:string};base:{research:string}}>}|undefined,checkpointErrors=0;
  vi.stubGlobal('fetch',vi.fn(async(url:URL,options:RequestInit)=>{
    if(String(url).endsWith('/session/checkpoint')){saved=JSON.parse(String(options.body));return {ok:true,json:async()=>({checkpointId:saved!.checkpointId,saved:true})};}
    if(String(url).endsWith('/session/checkpoint-error'))checkpointErrors++;
    return {ok:true,json:async()=>({})};
  }));
  function Editor(){const [title,setTitle]=useState('Saved revision r2');useDesktopDraft(key,'p1',{research:'r2'},{title},draft=>setTitle(draft.title));return <output>{title}</output>;}
  function Scene(){const [open,setOpen]=useState(false);return <><button onClick={()=>setOpen(true)}>Open research</button>{open&&<Editor/>}</>;}
  const container=document.createElement('div');document.body.append(container);const root=createRoot(container);
  try {
    await act(async()=>root.render(<NextIntlClientProvider locale="en" timeZone="UTC" messages={{}}><DesktopSession localRuntime><Scene/></DesktopSession></NextIntlClientProvider>));
    await act(async()=>window.labvizDesktopCheckpoint!('before-open'));
    expect(saved!.entries[key].value.title).toBe('Old unsaved draft');expect(saved!.entries[key].base.research).toBe('r1');
    expect(container.querySelector('[data-testid="desktop-recovery-warning"]')).toBeNull();
    await act(async()=>container.querySelector('button')!.click());
    expect(container.querySelector('output')?.textContent).toBe('Saved revision r2');
    expect(container.querySelector('[data-testid="desktop-recovery-warning"]')).not.toBeNull();
    await act(async()=>window.labvizDesktopCheckpoint!('blocked-conflict'));
    expect(checkpointErrors).toBe(1);expect(saved!.entries[key].value.title).toBe('Old unsaved draft');
    const acknowledge=[...container.querySelectorAll('button')].find(button=>button.textContent==='Use current saved state');expect(acknowledge).toBeDefined();
    await act(async()=>acknowledge!.click());await act(async()=>window.labvizDesktopCheckpoint!('accepted-current'));
    expect(saved!.entries[key].base.research).toBe('r2');expect(saved!.entries[key].value.title).toBe('Saved revision r2');
  } finally {
    await act(async()=>root.unmount());container.remove();vi.unstubAllGlobals();
    delete window.labvizDesktop;delete window.labvizDesktopTransport;delete window.labvizDesktopPending;delete window.labvizDesktopFlushPreferences;
  }
});
