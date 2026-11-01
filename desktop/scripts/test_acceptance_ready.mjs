import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const source=fs.readFileSync(new URL('../host/src/acceptance-ready.js',import.meta.url),'utf8');
function scenario(kind='figure') {
  const config={id:'test-only',sequence:1,origin:'http://127.0.0.1:54321',pathname:kind==='home'?'/':'/workspace/p1',profileId:'p1',kind,expectedTitle:'Unsaved figure'};
  const queue=[],signals=[];const location={origin:config.origin,pathname:config.pathname,set href(value){signals.push(new URL(value));}};
  const field=(value,shown=true)=>({value,isConnected:true,readOnly:false,disabled:false,labels:[{textContent:'Figure title'}],getClientRects:()=>shown?[{}]:[],matches(){return this.disabled;},getAttribute:()=>null});
  const hidden=field('Hidden quick plot',false),visible=field('Unsaved figure');
  const root={querySelector:selector=>selector==='button'?visible:{type:'file'}};
  const window={labvizDesktop:{profileId:'p1'},labvizDesktopCheckpoint:()=>{},__TAURI_INTERNALS__:{invoke:(name,{signal})=>{assert.equal(name,'acceptance_editor_ready');signals.push(signal);return Promise.resolve();}}};window.top=window;
  const document={visibilityState:'visible',readyState:'complete',hasFocus:()=>true,querySelector:()=>root,querySelectorAll:()=>[hidden,visible]};
  const context={window,location,document,URL,performance:{timeOrigin:100000,now:()=>1200},getComputedStyle:()=>({visibility:'visible'}),requestAnimationFrame:f=>queue.push(f)};
  vm.runInNewContext(source.replace('__LABVIZ_ACCEPTANCE_READY__',JSON.stringify(config)),context);
  const frames=(n=4)=>{for(let i=0;i<n;i++){const work=queue.splice(0);for(const run of work)run();}};
  return {signals,frames,window,document,visible};
}
let s=scenario();s.frames();assert.equal(s.signals.length,1);assert.equal(s.signals[0].rendererUtcSeconds,101.2);s.frames();assert.equal(s.signals.length,1);
s=scenario();s.visible.value='Wrong draft';s.frames();assert.equal(s.signals.length,0);s.visible.value='Unsaved figure';s.frames();assert.equal(s.signals.length,1);
s=scenario();s.visible.readOnly=true;s.frames();assert.equal(s.signals.length,0);
s=scenario();s.document.visibilityState='hidden';s.frames();assert.equal(s.signals.length,0);
s=scenario();s.window.labvizDesktopCheckpoint=undefined;s.frames();assert.equal(s.signals.length,0);
s=scenario();s.window.labvizDesktop.profileId='other';s.frames();assert.equal(s.signals.length,0);
s=scenario('home');s.visible.disabled=true;s.frames();assert.equal(s.signals.length,0);s.visible.disabled=false;s.frames();assert.equal(s.signals.length,1);
s=scenario('home');s.window.labvizDesktopCheckpoint=undefined;s.frames();assert.equal(s.signals.length,0);
console.log(JSON.stringify({passed:true,scenarios:8,scope:'Preinstalled native readiness observer; hidden duplicate, wrong draft, hydration, readonly, visibility, profile, home controls and single emission'}));
