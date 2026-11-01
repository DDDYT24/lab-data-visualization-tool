// Controlled headless faults only. This script refuses a native acceptance session.
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {randomUUID} from 'node:crypto';
import {observeCdpConnection} from '../scripts/cdp_observer.mjs';

const root=path.resolve(import.meta.dirname,'../..');
const {chromium,expect}=createRequire(path.join(root,'V2.0/web/package.json'))('@playwright/test');
const trial=path.resolve(process.argv[2]),action=process.argv[3];
if(!trial.startsWith(path.join(process.env.USERPROFILE,'LabVizD2')+path.sep))throw Error('Use an owned test trial');
const read=name=>JSON.parse(fs.readFileSync(path.join(trial,name)));
const config=read('session.json');
if(config.shell!=='headless-test-adapter')throw Error('Native application control is excluded');
if(!['client-disconnect','renderer-crash','host-close','planned-rebuild'].includes(action))throw Error('Invalid controlled fault');
const report={passed:false,action,nativeAppLaunched:false,scope:'Controlled owned headless browser; no desktop controls',historicalCauseConfirmed:false};
const command=action=>fs.writeFileSync(path.join(trial,'commands.json'),JSON.stringify({acceptanceId:config.id,id:randomUUID(),action}));
const events=()=>fs.readFileSync(path.join(trial,'headless-events.jsonl'),'utf8').split('\n').filter(Boolean).map(JSON.parse);
let observer;
try{
 const endpoint='http://127.0.0.1:'+config.debugPort;
 const browser=await chromium.connectOverCDP(endpoint),page=browser.contexts()[0].pages().find(p=>p.url().startsWith(read('services.json').webUrl));
 expect(page).toBeTruthy();
 await page.waitForRequest(request=>{
  if(new URL(request.url()).pathname!=='/session/activity')return false;
  const state=request.postDataJSON();return state.ready===true&&!state.busy&&state.pendingRequests===0;
 },{timeout:60000});
 const cycle=read('services.json').cycle;
 observer=observeCdpConnection(browser,page,{directory:path.join(trial,'cdp-observations'),label:action,cycleAtAttach:cycle,
  hostEventsPath:path.join(trial,'headless-events.jsonl'),readCurrentCycle:()=>read('services.json').cycle,
  probe:async()=>{
   const independent=await chromium.connectOverCDP(endpoint,{timeout:3000});
   try{
    const candidate=independent.contexts()[0].pages().find(p=>p.url().startsWith(read('services.json').webUrl)&&!p.isClosed());
    return {reachable:true,pageAlive:candidate?await Promise.race([candidate.evaluate(()=>Boolean(document.documentElement)),new Promise(resolve=>setTimeout(()=>resolve(false),1500))]):false};
   }finally{await independent.close();}
  }});
 if(action==='client-disconnect'){
  await browser.close();
 }else if(action==='renderer-crash'){
  const session=await browser.contexts()[0].newCDPSession(page);
  const crashed=page.waitForEvent('crash',{timeout:15000});
  void session.send('Page.crash').catch(()=>{});await crashed;
 }else if(action==='host-close'){
  const session=await browser.newBrowserCDPSession();
  void session.send('Browser.close').catch(()=>{});
  await expect.poll(()=>events().some(e=>e.event==='headless-browser-disconnected'&&e.expected===false),{timeout:15000}).toBe(true);
 }else{
  command('hide');
  await expect.poll(()=>events().some(e=>e.event==='services-suspended'),{timeout:120000}).toBe(true);
 }
 let actualFailure;
 try{
  await Promise.race([page.title(),new Promise((_,reject)=>setTimeout(()=>reject(new Error('Read-only probe timed out')),5000))]);
 }catch(error){actualFailure=error;}
 // Chromium may answer a later read after emitting a renderer-crash event.
 // The independently observed crash remains evidence; do not invent an error.
 if(action!=='renderer-crash')expect(actualFailure).toBeTruthy();
 const observation=await observer.capture(actualFailure);
 const expected={'client-disconnect':'client-transport-disconnected','renderer-crash':'renderer-crashed','host-close':'host-browser-closed','planned-rebuild':'planned-host-teardown'}[action];
 expect(observation.classification).toBe(expected);
 report.classification=observation.classification;report.errorKind=observation.errorKind;report.usedActualReadFailure=Boolean(actualFailure);report.rendererReadRecovered=action==='renderer-crash'&&!actualFailure;report.observation=path.relative(trial,observer.path);observer.dispose();
 if(action==='planned-rebuild'){
  command('show');await expect.poll(()=>read('services.json').cycle,{timeout:120000}).toBeGreaterThan(cycle);
  const fresh=await chromium.connectOverCDP(endpoint);
  const live=fresh.contexts()[0].pages().find(p=>p.url().startsWith(read('services.json').webUrl));expect(live).toBeTruthy();
  expect(await live.evaluate(()=>Boolean(document.documentElement))).toBe(true);await fresh.close();
  report.rebuildReopened=true;
 }
 report.passed=true;
}catch(error){report.error=String(error.stack??error);}
finally{
 observer?.dispose();command('exit');
 fs.writeFileSync(path.join(trial,'cdp-fault-result.json'),JSON.stringify(report,null,2));
}
console.log(JSON.stringify(report));process.exit(report.passed?0:1);
