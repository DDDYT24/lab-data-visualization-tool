import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { createHash, randomUUID } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';
import { spawnSync } from 'node:child_process';

const root = path.resolve(import.meta.dirname, '../..');
const require = createRequire(path.join(root, 'V2.0/web/package.json'));
const { chromium, expect } = require('@playwright/test');
const trial = path.resolve(process.argv[2]);
const action = process.argv[3] ?? 'roundtrip';
const config = JSON.parse(fs.readFileSync(path.join(trial, 'session.json')));
const read = name => JSON.parse(fs.readFileSync(path.join(trial, name)));
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const command = action => fs.writeFileSync(path.join(trial, 'commands.json'), JSON.stringify({acceptanceId:config.id,id:randomUUID(),action}));
const events = () => fs.readFileSync(read('profile/logs/desktop-host.json').hostLog,'utf8').trim().split('\n').map(JSON.parse);
const countEvent = name => events().filter(e=>e.event===name).length;
const report = {action,passed:false,checks:[],timings:{},scope:config.shell==='headless-test-adapter'?'Isolated headless production integration; no native window or tray acceptance':'Native desktop acceptance'};
function check(name,detail={}) { report.checks.push({name,...detail}); save(); console.log(name); }
function save() { fs.writeFileSync(path.join(trial,`suspend-${action}.json`),JSON.stringify(report,null,2)); }
function scientificCounts() {
  const db = new DatabaseSync(path.join(trial,'profile/appdata/LabVizV23/data/labviz.db'),{readOnly:true});
  db.exec('PRAGMA busy_timeout=10000');
  try {
    return {counts:Object.fromEntries(['projects','revisions','project_revisions','research_revisions','research_charts','research_figure_boards','research_templates','research_outputs'].filter(name=>db.prepare('SELECT name FROM sqlite_master WHERE name=?').get(name)).map(name=>[name,db.prepare(`SELECT count(*) AS n FROM ${name}`).get().n])),heads:db.prepare('SELECT project_id,revision_id FROM research_heads ORDER BY project_id').all(),projectHeads:db.prepare('SELECT id,current_revision_id FROM projects ORDER BY id').all()};
  } finally {db.close();}
}
async function connect() {
  const origin=read('services.json').webUrl;
  const browser=await chromium.connectOverCDP(`http://127.0.0.1:${config.debugPort}`);
  const page=browser.contexts()[0].pages().find(p=>p.url().startsWith(origin));
  expect(page).toBeTruthy();page.setDefaultTimeout(30000);
  return {page,browser,origin};
}
async function resume(previousCycle) {
  const started=Date.now();command('show');
  await expect.poll(()=>read('services.json').cycle,{timeout:180000,intervals:[50]}).toBeGreaterThan(previousCycle);
  let connected;
  await expect.poll(async()=>{try {connected=await connect();return !!connected;}catch{return false;}},{timeout:60000,intervals:[50]}).toBe(true);
  await expect(connected.page.getByTestId('research-workspace')).toBeVisible({timeout:60000});
  if(new URL(connected.page.url()).searchParams.get('researchTask')==='figures'){
    const snapshot=read('profile/session/checkpoint.json').payload;
    const project=new URL(connected.page.url()).pathname.split('/').at(-1);
    const expected=snapshot.entries[`figure:${project}`]?.value.title;
    if(expected!==undefined){await expect(connected.page.getByRole('textbox',{name:'Figure title',exact:true})).toHaveValue(expected);await expect(connected.page.getByRole('textbox',{name:'Figure title',exact:true})).toBeEnabled();}
  }
  report.timings.resumeMs=Date.now()-started;
  (report.timings.resumesMs??=[]).push(report.timings.resumeMs);
  return connected;
}
async function suspended(previous) {
  await expect.poll(()=>countEvent('services-suspended'),{timeout:180000}).toBeGreaterThan(previous);
  await pause(5000);
  const rows=fs.readFileSync(path.join(trial,'resources.jsonl'),'utf8').trim().split('\n').slice(-10).map(JSON.parse);
  report.suspendedPrivateCommitMiB=Math.max(...rows.map(r=>r.privateCommitBytes))/2**20;
  report.suspendedProcesses=rows.at(-1).processes.map(p=>({pid:p.pid,name:p.name,privateCommitMiB:p.privateCommitBytes/2**20}));
  expect(report.suspendedProcesses.some(p=>/node|webview|powershell/i.test(p.name))).toBe(false);
  expect(report.suspendedPrivateCommitMiB).toBeLessThanOrEqual(100);
  check(config.shell==='headless-test-adapter'?'headless-page-destroyed-and-services-stopped':'webview-destroyed-and-services-stopped', {privateCommitMiB:report.suspendedPrivateCommitMiB});
}

let page;
try {
  let connection=await connect();page=connection.page;
  const project=read('export-links.json')[0].href.match(/\/projects\/([^/]+)/)[1];
  if(action==='roundtrip') {
    await page.goto(connection.origin+`/workspace/${project}?workspace=research&researchTask=outputs`);
    await expect(page.getByTestId('research-workspace')).toBeVisible({timeout:60000});
    await page.waitForRequest(request=>{
      if(new URL(request.url()).pathname!=='/session/activity')return false;
      const state=request.postDataJSON();return state.ready===true&&state.busy===false&&state.pendingRequests===0;
    },{timeout:60000});
    const returnToTask=page.getByRole('button',{name:'Return to active task',exact:true});
    if(await returnToTask.isVisible())await returnToTask.click();
    // Recovery can restore the saved task and drawer after navigation.
    await page.getByRole('tab',{name:'Outputs',exact:true}).click();
    const reuse=page.getByTestId('reuse-workspace');
    await reuse.getByRole('combobox',{name:'Saved figure for template / output',exact:true}).click();
    await page.getByRole('option',{name:/D1 Two panels/}).first().click();
    await reuse.getByRole('button',{name:'Preview included output files',exact:true}).click();
    await reuse.getByRole('textbox',{name:'Figure legend (English)',exact:true}).fill('D3 retained output notes');
    await reuse.getByRole('textbox',{name:'Template name',exact:true}).fill('D3 unsaved template name');
    await page.getByRole('tab',{name:'Analysis',exact:true}).click();
    const charts=page.getByTestId('research-charts');
    await charts.getByRole('combobox',{name:'Saved charts',exact:true}).click();
    const selectedOption=page.getByRole('option',{name:/D1 Raw/}).first();
    const selectedChart=await selectedOption.getAttribute('data-value');
    expect(selectedChart).toBeTruthy();
    await selectedOption.click();
    await expect.poll(()=>charts.getByRole('combobox',{name:'Saved charts',exact:true}).evaluate(element=>element.parentElement.querySelector('input').value)).toBe(selectedChart);
    await charts.getByRole('textbox',{name:'Chart title',exact:true}).fill('D3 unsaved chart');
    await page.getByRole('tab',{name:'Data',exact:true}).click();
    const sources=page.getByRole('button',{name:'Data sources',exact:true});
    if(await sources.isVisible())await sources.click();
    const research=page.getByTestId('research-workspace');
    await research.getByLabel('Research source files',{exact:true}).setInputFiles(path.join(root,'outputs/desktop-d01/d0/fixtures/中文 路径/科研数据.csv'));
    await expect(research.getByRole('button',{name:'Preview preparation',exact:true})).toBeEnabled();
    for(const [label,field] of [['Measurement value','value'],['Sample ID','sample'],['Independent experimental unit','sample']]){
      await research.getByRole('combobox',{name:label,exact:true}).fill(field);
      await research.getByRole('combobox',{name:label,exact:true}).press('Tab');
    }
    if(await returnToTask.isVisible())await returnToTask.click();
    await page.getByRole('tab',{name:'Figures',exact:true}).click();
    const board=page.getByTestId('figure-workspace');
    const editor=board.getByText('Edit draft and panel layout',{exact:true});
    const figureTitle=board.getByRole('textbox',{name:'Figure title',exact:true});
    await figureTitle.waitFor({state:'attached',timeout:60000});
    if (!(await figureTitle.isVisible()) && await editor.isVisible()) await editor.click();
    await expect(figureTitle).toBeVisible({timeout:60000});
    await board.getByRole('textbox',{name:'Figure title',exact:true}).fill('D3 first unsaved title');
    await board.getByRole('textbox',{name:'Figure title',exact:true}).fill('D3 final unsaved title');
    await expect(board.getByRole('button',{name:'Undo layout edit',exact:true})).toBeEnabled();
    const before=scientificCounts();const old=read('services.json');const sleeps=countEvent('services-suspended');
    command('hide');await suspended(sleeps);
    const envelope=read('profile/session/checkpoint.json');
    expect(envelope.payload.profileId).toBe(old.profileId);
    expect(envelope.payload.entries[`figure:${project}`].value.title).toBe('D3 final unsaved title');
    expect(envelope.payload.entries[`chart:${project}`].value.selectedChartId).toBe(selectedChart);
    expect(envelope.payload.entries[`figure:${project}`].value.undoDrafts.length).toBeGreaterThan(0);
    const file=envelope.payload.entries[`research:${project}`].value.files[0];
    expect(file.__labvizFile).toBeTruthy();
    expect(createHash('sha256').update(fs.readFileSync(path.join(trial,'profile/session/files',file.__labvizFile))).digest('hex')).toBe(file.__labvizFile);
    expect(envelope.payload.entries[`research:${project}`].value.preview).toBeUndefined();
    expect(envelope.payload.entries[`reuse:${project}`].value.outputCheck).toBeUndefined();
    check('durable-checkpoint-and-owned-staged-source');
    const controllerConfig=read('controller-session.json');
    const payload=controllerConfig.payloadRoot;
    const access=fs.readFileSync(path.join(trial,'profile/appdata/LabVizV23/access.dpapi'));
    const duplicate=spawnSync(path.join(payload,'runtime/python/python.exe'),['-B','-X','utf8',path.join(payload,'../runtime/controller.py'),path.join(trial,'controller-session.json')],{windowsHide:true,encoding:'utf8',timeout:15000});
    expect(duplicate.status).toBe(1);expect(duplicate.stdout).not.toContain('"type": "ready"');
    expect(fs.readFileSync(path.join(trial,'profile/appdata/LabVizV23/access.dpapi'))).toEqual(access);
    check('sleeping-profile-refuses-a-second-controller-before-credential-rotation');
    connection=await resume(old.cycle);page=connection.page;
    expect(read('services.json').profileId).toBe(old.profileId);expect(read('services.json').api.pid).not.toBe(old.api.pid);
    await expect(page.getByRole('textbox',{name:'Figure title',exact:true})).toHaveValue('D3 final unsaved title');
    await page.getByRole('button',{name:'Undo layout edit',exact:true}).click();
    await expect(page.getByRole('textbox',{name:'Figure title',exact:true})).not.toHaveValue('D3 final unsaved title');
    await page.getByRole('button',{name:'Redo layout edit',exact:true}).click();
    await expect(page.getByRole('textbox',{name:'Figure title',exact:true})).toHaveValue('D3 final unsaved title');
    await page.getByRole('tab',{name:'Analysis',exact:true}).click();
    await expect(page.getByTestId('research-charts').getByRole('textbox',{name:'Chart title',exact:true})).toHaveValue('D3 unsaved chart');
    await expect.poll(()=>page.getByTestId('research-charts').getByRole('combobox',{name:'Saved charts',exact:true}).evaluate(element=>element.parentElement.querySelector('input').value)).toBe(selectedChart);
    await page.getByRole('tab',{name:'Outputs',exact:true}).click();
    await expect(page.getByRole('textbox',{name:'Template name',exact:true})).toHaveValue('D3 unsaved template name');
    await page.getByRole('button',{name:'Preview included output files',exact:true}).click();
    await expect(page.getByRole('textbox',{name:'Figure legend (English)',exact:true})).toHaveValue('D3 retained output notes');
    await page.getByRole('tab',{name:'Data',exact:true}).click();
    await page.getByRole('button',{name:'Preview preparation',exact:true}).click();
    await expect(page.getByTestId('research-counts')).toContainText('Observations: 12');
    expect(scientificCounts()).toEqual(before);
    check('actual-recreation-restores-drafts-files-undo-notes-and-selected-chart-without-scientific-revisions');
    await page.getByRole('tab',{name:'Figures',exact:true}).click();
    const link=read('export-links.json').find(e=>e.format==='pdf');
    const bytes=await (await page.request.get(new URL(link.href,connection.origin).toString())).body();
    expect(createHash('sha256').update(bytes).digest('hex')).toBe(link.sha256);
    check('frozen-export-unchanged-after-service-restart');
  } else if(action==='cancel') {
    const old=read('services.json');const requests=countEvent('checkpoint-requested');const sleeps=countEvent('services-suspended');
    await page.evaluate(()=>{window.__heldDesktopCheckpoint=window.labvizDesktopCheckpoint;window.labvizDesktopCheckpoint=async()=>{};});
    command('hide');
    await expect.poll(()=>countEvent('checkpoint-requested'),{timeout:60000}).toBeGreaterThan(requests);
    command('show');await expect.poll(()=>countEvent('suspension-cancelled'),{timeout:15000}).toBeGreaterThan(0);
    await pause(2000);expect(read('services.json').api.pid).toBe(old.api.pid);expect(countEvent('services-suspended')).toBe(sleeps);
    await page.evaluate(()=>{window.labvizDesktopCheckpoint=window.__heldDesktopCheckpoint;delete window.__heldDesktopCheckpoint;});
    check('show-cancels-checkpoint-and-drain-without-restarting-services');
  } else if(action==='blocked') {
    const old=read('services.json');const blocked=countEvent('suspension-blocked');const sleeps=countEvent('services-suspended');
    await page.evaluate(()=>{window.__heldDesktopCheckpoint=window.labvizDesktopCheckpoint;window.labvizDesktopCheckpoint=async id=>{const p=window.labvizDesktopTransport;await fetch(new URL('/session/checkpoint-error',p.url),{method:'POST',headers:{'Content-Type':'application/json','X-LabViz-Desktop':p.token},body:JSON.stringify({checkpointId:id,message:'Acceptance: disk write failed'})});};});
    command('hide');await expect.poll(()=>countEvent('suspension-blocked'),{timeout:60000}).toBeGreaterThan(blocked);
    command('show');await expect(page.getByTestId('desktop-suspend-warning')).toBeVisible();
    expect(read('services.json').api.pid).toBe(old.api.pid);expect(countEvent('services-suspended')).toBe(sleeps);
    await page.evaluate(()=>{window.labvizDesktopCheckpoint=window.__heldDesktopCheckpoint;delete window.__heldDesktopCheckpoint;});
    check('save-failure-keeps-editor-and-services-and-reopens-admission');
  } else if(action==='download-guards') {
    const sleeps=countEvent('services-suspended');const old=read('services.json');
    command('download-start:a');await pause(300);command('download-start:b');await pause(300);command('hide');
    await pause((config.idleSeconds+2)*1000);expect(countEvent('services-suspended')).toBe(sleeps);
    command('download-end:a');await pause(300);command('download-end:a');
    await pause((config.idleSeconds+2)*1000);expect(countEvent('services-suspended')).toBe(sleeps);
    command('download-end:b');await suspended(sleeps);connection=await resume(old.cycle);page=connection.page;
    check('duplicate-completion-cannot-release-another-download-guard');
  } else if(action==='corrupt') {
    const original=fs.readFileSync(path.join(trial,'profile/session/checkpoint.json'));
    expect(original.toString()).toBe('{acceptance-corrupt-checkpoint');
    const before=scientificCounts();
    await expect(page.getByTestId('desktop-recovery-warning')).toBeVisible();
    expect(fs.readFileSync(path.join(trial,'profile/session/checkpoint.json'))).toEqual(original);
    await page.getByRole('button',{name:'Use current saved state',exact:true}).click();
    await expect(page.getByTestId('desktop-recovery-warning')).toHaveCount(0);
    expect(scientificCounts()).toEqual(before);
    check('corrupt-checkpoint-is-retained-and-saved-state-requires-explicit-choice');
  } else if(action==='repeat') {
    const before=scientificCounts();
    for(let i=0;i<2;i++) {
      const old=read('services.json');const sleeps=countEvent('services-suspended');command('hide');await suspended(sleeps);
      connection=await resume(old.cycle);page=connection.page;
      await expect(page.getByRole('textbox',{name:'Figure title',exact:true})).toHaveValue('D3 final unsaved title');
      expect(scientificCounts()).toEqual(before);
      check('repeated-suspend-resume-without-duplicate-work',{iteration:i+1,cycle:read('services.json').cycle});
    }
  }
  report.passed=true;save();console.log(JSON.stringify({passed:true,checks:report.checks.length,timings:report.timings}));process.exit(0);
} catch(error) {
  report.error=String(error.stack??error);save();if(page){try{await page.screenshot({path:path.join(trial,`suspend-${action}-failure.png`)});}catch{}}
  console.error(report.error);process.exit(1);
}
