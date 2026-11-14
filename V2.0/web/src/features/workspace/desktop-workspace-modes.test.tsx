// @vitest-environment jsdom
import React, {act} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, beforeEach, expect, it, vi} from 'vitest';
import {WorkspaceScreen} from './workspace-screen';
import {useWorkspaceStore} from './workspace-store';

const state=vi.hoisted(()=>({research:true,mounts:[] as string[],job:null as {stage:string}|null}));
vi.mock('next-intl',()=>({useLocale:()=> 'en',useTranslations:()=> (key:string)=>key}));
vi.mock('next/dynamic',async()=>{
  const React=await import('react');
  return {default:(load:()=>Promise<React.ComponentType>)=>{
    const Component=React.lazy(async()=>({default:await load()}));
    return function Dynamic(props:object){return <React.Suspense><Component {...props}/></React.Suspense>;};
  }};
});
vi.mock('./use-workspace-project',()=>({useWorkspaceProject:()=>({job:state.job,loadStatus:'ready',preview:{},quality:{},uploading:false})}));
vi.mock('@/components/common/workflow-stepper',()=>({WorkflowStepper:()=>null}));
vi.mock('./desktop-session',async()=>{
  const React=await import('react');
  return {useDesktopDraft:(key:string,_project:unknown,_base:unknown,_value:unknown,restore:(value:unknown)=>void,ready:boolean)=>{
    const done=React.useRef(false);
    React.useEffect(()=>{
      if(key.startsWith('navigation:')&&ready&&!done.current){
        done.current=true;
        restore({researchTask:'figures',researchMode:state.research,researchOpened:true,quickPlotOpened:true});
      }
    },[key,ready,restore]);
  }};
});
vi.mock('./chart-step',async()=>{
  const React=await import('react');
  return {ChartStep:function Chart(){React.useEffect(()=>{state.mounts.push('quick');},[]);return <div data-testid="quick-editor"/>;}};
});
vi.mock('./research-workspace',async()=>{
  const React=await import('react');
  return {ResearchWorkspace:function Research(){React.useEffect(()=>{state.mounts.push('research');},[]);return <div data-testid="research-editor"/>;}};
});
let root:Root|undefined,container:HTMLDivElement;
(globalThis as typeof globalThis & {IS_REACT_ACT_ENVIRONMENT:boolean}).IS_REACT_ACT_ENVIRONMENT=true;
beforeEach(()=>{state.mounts=[];state.job=null;useWorkspaceStore.setState({projectId:'p1',currentRevisionId:'r1',currentStep:'chart'});});
afterEach(async()=>{if(root)await act(async()=>root!.unmount());root=undefined;container?.remove();});
async function render(research:boolean){
  state.research=research;container=document.createElement('div');document.body.append(container);root=createRoot(container);
  await act(async()=>{root!.render(<WorkspaceScreen initialProjectId="p1" initialResearchMode={research}/>);});
}
async function click(text:string){const b=[...container.querySelectorAll('button')].find(b=>b.textContent===text);expect(b).toBeDefined();await act(async()=>b!.click());}
it('does not mount the parked quick plot during research recovery, but opens it on request',async()=>{
  await render(true);expect(state.mounts).toEqual(['research']);expect(container.querySelector('[data-testid="quick-editor"]')).toBeNull();
  await click('Back to quick plot');expect(state.mounts).toEqual(['research','quick']);
  await click('Research workspace');expect(state.mounts).toEqual(['research','quick']);
});
it('does not mount parked research during quick plot recovery, but preserves it after opening',async()=>{
  await render(false);expect(state.mounts).toEqual(['quick']);expect(container.querySelector('[data-testid="research-editor"]')).toBeNull();
  await click('Research workspace');expect(state.mounts).toEqual(['quick','research']);
  await click('Back to quick plot');expect(state.mounts).toEqual(['quick','research']);
});
it('mounts research only after the upload job has ended',async()=>{
  state.job={stage:'parsing'};await render(true);expect(state.mounts).toEqual([]);
  state.job={stage:'ready'};await act(async()=>{root!.render(<WorkspaceScreen initialProjectId="p1" initialResearchMode/>);});
  expect(state.mounts).toEqual(['research']);
});
