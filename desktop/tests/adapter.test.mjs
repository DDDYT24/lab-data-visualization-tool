import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../runtime/adapter.js',import.meta.url),'utf8');
const key='labviz:user-preferences:v1';
function execute(storage,generation,origin='http://127.0.0.1:12345',writable=true){
  const config={origin:'http://127.0.0.1:12345',profileId:'p',preferences:{generation,url:'http://127.0.0.1:55555/preferences',token:'synthetic-private-token',initial:{locale:'en',storage:{[key]:'dark','labviz:v23:interface-mode:v1':'overview'}}}};
  const location={origin,href:origin+'/',hash:''};
  const window={location,fetch:async()=>({ok:writable}),addEventListener(){}};
  vm.runInNewContext(source.replace('__LABVIZ_DESKTOP_CONFIGURATION__',JSON.stringify(config)),{window,location,URL,AbortSignal,setTimeout,localStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>storage.set(k,v)},document:{cookie:'labviz-locale=en',documentElement:{lang:'en'}},setInterval(){},queueMicrotask(){}});
  return window;
}
test('a reused port restores the latest profile preferences for the new service generation',()=>{
  const storage=new Map([['labviz:desktop-profile:v1','p:old'],[key,'light']]);
  execute(storage,'new');assert.equal(storage.get(key),'dark');assert.equal(storage.get('labviz:v23:interface-mode:v1'),'overview');
  assert.ok([...storage.values()].every(value=>!value.includes('synthetic-private-token')));
});
test('full navigation in the same service generation retains current edits to preferences',()=>{
  const storage=new Map();execute(storage,'same');storage.set(key,'light');execute(storage,'same');assert.equal(storage.get(key),'light');
});
test('an unrelated origin receives no desktop bridge or profile storage changes',()=>{
  const storage=new Map();const window=execute(storage,'new','https://example.test');assert.equal(window.labvizDesktop,undefined);assert.equal(storage.size,0);
});
test('checkpoint preference flush requires a successful durable-storage response',async()=>{
  await execute(new Map(),'ready').labvizDesktopFlushPreferences();
  await assert.rejects(execute(new Map(),'read-only','http://127.0.0.1:12345',false).labvizDesktopFlushPreferences(),/Preferences could not be saved/);
});
