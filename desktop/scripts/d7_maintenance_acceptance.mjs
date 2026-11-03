// Headless production maintenance and ordinary-suspension race acceptance.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { createHash, randomUUID } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';
import { pathToFileURL } from 'node:url';
const root=path.resolve(import.meta.dirname,'../..');
const {observeCdpConnection}=await import(pathToFileURL(path.join(root,'desktop/scripts/cdp_observer.mjs')).href);
const require=createRequire(path.join(root,'V2.0/web/package.json'));
const {chromium,expect}=require('@playwright/test');
const trial=path.resolve(process.argv[2]),bundle=path.resolve(process.argv[3]);
const read=name=>JSON.parse(fs.readFileSync(path.join(trial,name)));
const config=read('session.json');
const report={passed:false,packageId:JSON.parse(fs.readFileSync(path.join(bundle,'desktop-package.json'))).packageId,
  scope:'Headless production UI and actual owned services; no native controls',checks:[]};
const save=()=>fs.writeFileSync(path.join(trial,'d7-maintenance.json'),JSON.stringify(report,null,2));
function check(name,values={}){report.checks.push({name,...values});save();console.log(name)}
const command=action=>fs.writeFileSync(path.join(trial,'commands.json'),JSON.stringify({acceptanceId:config.id,id:randomUUID(),action}));
const events=()=>fs.readFileSync(path.join(trial,'headless-events.jsonl'),'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
const count=name=>events().filter(e=>e.event===name).length;
let observer;
function scientific(){
 const db=new DatabaseSync(path.join(trial,'profile/appdata/LabVizV23/data/labviz.db'),{readOnly:true});
 try{
  db.exec('PRAGMA busy_timeout=10000');
  return Object.fromEntries(['research_heads','research_revisions','research_charts','research_figure_boards','research_figure_jobs','research_figure_artifacts','research_templates','research_outputs'].map(table=>{
    const rows=db.prepare(`SELECT * FROM ${table} ORDER BY rowid`).all();
    return [table,createHash('sha256').update(JSON.stringify(rows)).digest('hex')];
  }));
 }finally{db.close()}
}
async function connect(){
 const browser=await chromium.connectOverCDP('http://127.0.0.1:'+config.debugPort),origin=read('services.json').webUrl;
 const page=browser.contexts()[0].pages().find(p=>p.url().startsWith(origin));if(!page)throw Error('Production page unavailable');page.setDefaultTimeout(30000);
 observer?.dispose();
 observer=observeCdpConnection(browser,page,{directory:path.join(trial,'cdp-observations'),label:'maintenance',
  cycleAtAttach:read('services.json').cycle,hostEventsPath:path.join(trial,'headless-events.jsonl'),readCurrentCycle:()=>read('services.json').cycle,
  readOperation:()=>read('profile/maintenance.json'),
  probe:async()=>{
   const independent=await chromium.connectOverCDP('http://127.0.0.1:'+config.debugPort,{timeout:5000});
   try{
    const candidate=independent.contexts()[0].pages().find(value=>value.url().startsWith(read('services.json').webUrl)&&!value.isClosed());
    const responsive=candidate?await Promise.race([candidate.evaluate(()=>Boolean(document.documentElement)),new Promise(resolve=>setTimeout(()=>resolve(false),3000))]):false;
    return {reachable:true,pageAlive:responsive===true};
   }
   finally{await independent.close();}
  }});
 return{browser,page,origin};
}
async function recreated(cycle){
 await expect.poll(()=>read('services.json').cycle,{timeout:180000}).toBeGreaterThan(cycle);
 let result;await expect.poll(async()=>{try{result=await connect();return true}catch{return false}},{timeout:60000}).toBe(true);
 await expect(result.page.getByTestId('desktop-data-tools')).toBeVisible({timeout:60000});return result;
}
try{
 let connection=await connect(),page=connection.page;
 const project=read('export-links.json')[0].href.match(/\/projects\/([^/]+)/)[1];
 await page.goto(connection.origin+`/workspace/${project}?workspace=research&researchTask=analysis`);
 await expect(page.getByTestId('research-workspace')).toBeVisible({timeout:60000});
 await page.waitForRequest(request=>{
  if(new URL(request.url()).pathname!=='/session/activity')return false;
  const state=request.postDataJSON();return state.ready===true&&state.busy===false&&state.pendingRequests===0;
 },{timeout:60000});
 const returnToTask=page.getByRole('button',{name:'Return to active task',exact:true});if(await returnToTask.isVisible())await returnToTask.click();
 await page.getByRole('tab',{name:'Analysis',exact:true}).click();
 const title=page.getByTestId('research-charts').getByRole('textbox',{name:'Chart title',exact:true});
 await expect(title).toBeEnabled({timeout:60000});await title.fill('D7 retained during cache maintenance');
 await page.getByRole('button',{name:'Save and computation details',exact:true}).click();
 await expect(page.getByTestId('desktop-data-tools')).toBeVisible();
 const before=scientific(),cycle=read('services.json').cycle;
 const cache=path.join(trial,'profile/appdata/LabVizV23/cache/matplotlib');fs.mkdirSync(cache,{recursive:true});
 const probe=path.join(cache,'d7-regenerable-probe.json');fs.writeFileSync(probe,'regenerable cache probe');
 await page.getByRole('button',{name:'Clear calculation cache',exact:true}).click();
 connection=await recreated(cycle);page=connection.page;
 const result=read('profile/maintenance.json');expect(result.action).toBe('clear-cache');expect(result.state).toBe('completed');
 expect(result.removedFiles).toBeGreaterThan(0);expect(fs.existsSync(probe)).toBe(false);expect(scientific()).toEqual(before);
 expect(read('profile/session/checkpoint.json').payload.entries[`chart:${project}`].value.title).toBe('D7 retained during cache maintenance');
 check('real-cache-button-checkpoints-stops-cleans-and-reopens',{removedFiles:result.removedFiles,removedBytes:result.removedBytes});
 for(const artifact of [...read('workflow-full.json').exports,...read('workflow-large.json').exports]){
  const response=await page.request.get(new URL(artifact.href,connection.origin).toString());expect(response.ok()).toBe(true);
  expect(createHash('sha256').update(await response.body()).digest('hex')).toBe(artifact.sha256);
 }
 check('cache-maintenance-keeps-all-24-frozen-downloads');
 // Hold the first ordinary checkpoint before it ACKs; the next maintenance
 // checkpoint uses the original implementation without a hold.
 await page.evaluate(()=>{
  const original=window.labvizDesktopCheckpoint;
  window.labvizDesktopCheckpoint=async id=>{
   if(!window.__d7HeldCheckpoint){window.__d7HeldCheckpoint={id,release:()=>original(id)};return;}
   return original(id);
  };
 });
 const cancelled=count('suspension-cancelled'),previous=read('services.json').cycle;
 command('hide');
 await expect.poll(()=>page.evaluate(()=>window.__d7HeldCheckpoint?.id),{timeout:60000}).toBeTruthy();
 const held=await page.evaluate(()=>window.__d7HeldCheckpoint.id);
 const admitted=await page.evaluate(async()=>{
  const transport=window.labvizDesktopTransport;
  const response=await fetch(new URL('/maintenance/submit',transport.url),{method:'POST',headers:{'Content-Type':'application/json','X-LabViz-Desktop':transport.token},body:JSON.stringify({action:'backup'})});
  return{status:response.status,body:await response.json()};
 });
 expect(admitted.status).toBe(202);
 await page.evaluate(()=>window.__d7HeldCheckpoint.release());
 connection=await recreated(previous);page=connection.page;
 const operation=read('profile/maintenance.json');expect(operation.requestId).toBe(admitted.body.requestId);expect(operation.state).toBe('completed');
 expect(count('suspension-cancelled')).toBeGreaterThan(cancelled);
 const backedUp=JSON.parse(fs.readFileSync(path.join(operation.backupDirectory,'desktop/session/checkpoint.json'))).payload;
 expect(backedUp.checkpointId).not.toBe(held);expect(backedUp.entries[`chart:${project}`].value.title).toBe('D7 retained during cache maintenance');
 expect(scientific()).toEqual(before);
 check('late-maintenance-cancels-ordinary-checkpoint-and-captures-a-fresh-backup',{backupId:operation.backupId,oldCheckpointId:held,newCheckpointId:backedUp.checkpointId});
 command('show');observer?.dispose();report.passed=true;save();console.log(JSON.stringify({passed:true,checks:report.checks.length}));process.exit(0);
}catch(error){
 report.error=String(error.stack??error);
 try{if(observer){const observation=await observer.capture(error);report.cdpFailure={classification:observation.classification,path:observer.path};observer.dispose();}}
 catch{report.cdpDiagnosticCaptureFailed=true;}
 save();console.error(report.error);process.exit(1)
}
