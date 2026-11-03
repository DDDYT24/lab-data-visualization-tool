// @vitest-environment jsdom
import React, {act, useState} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {NextIntlClientProvider} from 'next-intl';
import {afterEach, beforeEach, expect, it, vi} from 'vitest';
import {DesktopSession, useDesktopDraft, forgetDesktopProject} from './desktop-session';
import {ResearchChartWorkspace} from './research-chart-workspace';

const apiState=vi.hoisted(()=>({list:vi.fn(async()=>({charts:[]})),get:vi.fn(async(...args:unknown[])=>{void args;return null as unknown;})}));
vi.mock('@/lib/api/research-chart-api',()=>({researchChartApi:{list:apiState.list,get:apiState.get,artifact:()=>'/artifact'}}));
type Entry={base:{research:string};value:{title:string}};
let root:Root|undefined,container:HTMLDivElement,captured:{entries:Record<string,Entry>};
let profileSequence=0;
const key='chart:p1';
(globalThis as typeof globalThis & {IS_REACT_ACT_ENVIRONMENT:boolean}).IS_REACT_ACT_ENVIRONMENT=true;
function Editor({revision,project}:{revision:string;project:string}){
 const [title,setTitle]=useState('Saved chart title');
 useDesktopDraft('chart:'+project,project,{research:revision},{title},d=>setTitle(d.title));
 return <><output>{title}</output><button id="edit" onClick={()=>setTitle('UNSAVED title')}>Edit</button></>;
}
function Scene({actual=false}:{actual?:boolean}){
 const [open,setOpen]=useState(true),[revision,setRevision]=useState('r1'),[project,setProject]=useState('p1');
 return <><button id="navigate" onClick={()=>setOpen(!open)}>Navigate</button><button id="switch" onClick={()=>setProject(project==='p1'?'p2':'p1')}>Switch</button><button id="revision" onClick={()=>setRevision('r2')}>Revision</button>{open?(actual?<ResearchChartWorkspace key={project} projectId={project} revisionId={revision} pending={false}/>:<Editor key={project} project={project} revision={revision}/>):<p>Home</p>}</>;
}
async function setup(actual=false,initial:Record<string,unknown>|null=null){
 const profileId='navigation-test-'+(++profileSequence);
 Object.defineProperty(window,'labvizDesktop',{value:{profileId},configurable:true});
 Object.defineProperty(window,'labvizDesktopTransport',{value:{url:'http://127.0.0.1:52525/',token:'test-only',session:{initial:initial?{...initial,profileId}:null,warning:null}},configurable:true});
 window.labvizDesktopPending=()=>0;window.labvizDesktopFlushPreferences=async()=>{};
 vi.stubGlobal('fetch',vi.fn(async(url:URL,options:RequestInit)=>{
  if(String(url).endsWith('/session/checkpoint')){const value=JSON.parse(String(options.body));captured=value;return {ok:true,json:async()=>({checkpointId:value.checkpointId,saved:true})};}
  return {ok:true,json:async()=>({})};
 }));
 container=document.createElement('div');document.body.append(container);root=createRoot(container);
 await act(async()=>root!.render(<NextIntlClientProvider locale="en" timeZone="UTC" messages={{}}><DesktopSession localRuntime><Scene actual={actual}/></DesktopSession></NextIntlClientProvider>));
}
async function click(id:string){await act(async()=>container.querySelector<HTMLButtonElement>(id)!.click());}
async function save(){await act(async()=>window.labvizDesktopCheckpoint!('test-checkpoint'));}
beforeEach(()=>{vi.useRealTimers();apiState.list.mockResolvedValue({charts:[]});apiState.get.mockResolvedValue(null);apiState.get.mockClear();});
afterEach(async()=>{if(root)await act(async()=>root!.unmount());root=undefined;container?.remove();vi.unstubAllGlobals();});
it('preserves a draft and checkpoint across normal editor unmount and return',async()=>{
 await setup();await click('#edit');await save();expect(captured.entries[key].value.title).toBe('UNSAVED title');
 await click('#navigate');await save();expect(captured.entries[key].value.title).toBe('UNSAVED title');
 await click('#navigate');expect(container.querySelector('output')?.textContent).toBe('UNSAVED title');
 await new Promise(r=>setTimeout(r,260));await save();expect(captured.entries[key].value.title).toBe('UNSAVED title');
});
it('restores the actual scientific chart title after navigation',async()=>{
 await setup(true);const input=container.querySelector<HTMLInputElement>('input[maxlength="160"]')!;
 await act(async()=>{Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value')!.set!.call(input,'Actual unsaved title');input.dispatchEvent(new Event('input',{bubbles:true}));});
 await click('#navigate');await click('#navigate');
 expect(container.querySelector<HTMLInputElement>('input[maxlength="160"]')!.value).toBe('Actual unsaved title');
 await new Promise(r=>setTimeout(r,260));await save();expect(captured.entries[key].value.title).toBe('Actual unsaved title');
});
it('retains the old draft and requires a decision if the saved revision changed',async()=>{
 await setup();await click('#edit');await click('#navigate');await click('#revision');await click('#navigate');
 expect(container.querySelector('output')?.textContent).toBe('Saved chart title');
 expect(container.querySelector('[data-testid="desktop-recovery-warning"]')).not.toBeNull();
 const resolve=Array.from(container.querySelectorAll('button')).find(b=>b.textContent==='Use current saved state')!;
 await act(async()=>resolve.click());await save();expect(captured.entries[key].base.research).toBe('r2');expect(captured.entries[key].value.title).toBe('Saved chart title');
});

it('keeps different projects isolated when switching away and back',async()=>{
 await setup();await click('#edit');await click('#switch');
 expect(container.querySelector('output')?.textContent).toBe('Saved chart title');
 await click('#switch');expect(container.querySelector('output')?.textContent).toBe('UNSAVED title');
 await new Promise(r=>setTimeout(r,260));await save();expect(captured.entries['chart:p1'].value.title).toBe('UNSAVED title');
 expect(captured.entries['chart:p2'].value.title).toBe('Saved chart title');
});
it('forgets only the explicitly deleted project and allows later checkpointing',async()=>{
 await setup();await click('#edit');await click('#switch');await click('#navigate');
 await act(async()=>forgetDesktopProject('p1'));await save();
 expect(captured.entries['chart:p1']).toBeUndefined();expect(captured.entries['chart:p2']).toBeDefined();
});
it('reloads saved chart pagination from the service instead of serializing result rows',async()=>{
 const chartId='chart-offset';
 apiState.get.mockResolvedValue({id:chartId,revisionId:'r1',spec:{kind:'grouped_strip',title:'Saved offset',pairTimes:[],jitterSeed:'test',summary:'none',summaryUnit:'experimental_unit'},observations:300,excluded:0,groups:[],densityMethods:[],rows:[],sources:[],pairs:[],pairCount:0,offset:100,totalRows:300,resultSha256:'0123456789abcdef'});
 await setup(true,{format:'labviz-desktop-checkpoint-v1',route:'/workspace/p1',checkpointId:'old',entries:{'chart:p1':{version:1,projectId:'p1',base:{research:'r1'},value:{pointRendering:'auto',kind:'grouped_strip',title:'Unsaved offset',seed:'test',summary:'none',summaryUnit:'experimental_unit',first:'before',second:'after',bandwidth:'',scale:'equal_width',points:true,selectedChartId:chartId,selectedChartOffset:100}}}});
 expect(apiState.get).toHaveBeenCalledWith('p1',chartId,100);
 expect(container.querySelector<HTMLInputElement>('input[maxlength="160"]')!.value).toBe('Unsaved offset');
 await new Promise(r=>setTimeout(r,260));await save();
 expect(captured.entries[key].value).not.toHaveProperty('rows');
 expect(captured.entries[key].value).toHaveProperty('selectedChartOffset',100);
});
