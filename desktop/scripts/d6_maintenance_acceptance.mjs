// Private headless integration. No native window or personal browser is opened.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { DatabaseSync } from 'node:sqlite';
const root=path.resolve(import.meta.dirname,'../..');
const require=createRequire(path.join(root,'V2.0/web/package.json'));
const {chromium,expect}=require('@playwright/test');
const trial=path.resolve(process.argv[2]);
const read=name=>JSON.parse(fs.readFileSync(path.join(trial,name)));
const config=read('session.json');
const bundle=path.join(root,'outputs/desktop-d06/bundle');
const report={passed:false,scope:'Private headless production UI and background processes only; native desktop not controlled',checks:[]};
const save=()=>fs.writeFileSync(path.join(trial,'d6-maintenance.json'),JSON.stringify(report,null,2));
function check(name,values={}){report.checks.push({name,...values});save();console.log(name)}
function scientific(profile=path.join(trial,'profile')){
 const db=new DatabaseSync(path.join(profile,'appdata/LabVizV23/data/labviz.db'),{readOnly:true});
 db.exec('PRAGMA busy_timeout=10000');
 try{return {counts:Object.fromEntries(['projects','revisions','research_revisions','research_charts','research_figure_boards','research_templates','research_outputs'].filter(n=>db.prepare('SELECT name FROM sqlite_master WHERE name=?').get(n)).map(n=>[n,db.prepare(`SELECT count(*) AS n FROM ${n}`).get().n])),heads:db.prepare('SELECT project_id,revision_id FROM research_heads ORDER BY project_id').all()}}
 finally{db.close()}
}
async function connect(){
 const browser=await chromium.connectOverCDP('http://127.0.0.1:'+config.debugPort);const origin=read('services.json').webUrl;
 const page=browser.contexts()[0].pages().find(p=>p.url().startsWith(origin));if(!page)throw Error('Production page unavailable');page.setDefaultTimeout(30000);return{browser,page,origin};
}
async function recreated(previous){
 await expect.poll(()=>read('services.json').cycle,{timeout:180000}).toBeGreaterThan(previous);
 let connection;await expect.poll(async()=>{try{connection=await connect();return true}catch{return false}},{timeout:60000}).toBe(true);
 await expect(connection.page.getByTestId('desktop-data-tools')).toBeVisible({timeout:60000});return connection;
}
let page;
try{
 let connection=await connect();page=connection.page;
 const project=read('export-links.json')[0].href.match(/\/projects\/([^/]+)/)[1];
 await page.goto(connection.origin+`/workspace/${project}?workspace=research&researchTask=analysis`);
 await expect(page.getByTestId('research-charts').locator('input').first()).toBeAttached({timeout:60000});
 await page.getByRole('tab',{name:'Analysis',exact:true}).click();
 const drawerReturn=page.getByRole('button',{name:'Return to active task',exact:true});if(await drawerReturn.isVisible())await drawerReturn.click();
 await expect(page.getByTestId('research-charts').getByRole('textbox',{name:'Chart title',exact:true})).toBeEnabled({timeout:30000});
 await page.getByTestId('research-charts').getByRole('textbox',{name:'Chart title',exact:true}).fill('D6 retained after maintenance');
 await page.getByRole('button',{name:'Preview PDF',exact:true}).click();
 const iframe=page.getByTitle('Saved PDF preview',{exact:true});await expect(iframe).toBeVisible();
 const pdfUrl=await iframe.getAttribute('src');const pdf=await page.request.get(new URL(pdfUrl,connection.origin).toString());
 expect(pdf.headers()['content-type']).toContain('application/pdf');expect((await pdf.body()).subarray(0,4).toString()).toBe('%PDF');
 await page.getByRole('button',{name:'Return to workspace',exact:true}).click();
 check('bounded-inline-frozen-pdf-url-and-bytes',{nativeRenderingAccepted:false});
 await page.getByRole('button',{name:'Save and computation details',exact:true}).click();
 await expect(page.getByTestId('desktop-data-tools')).toBeVisible();
 const before=scientific();const oldCycle=read('services.json').cycle;
 await page.getByRole('button',{name:'Back up data and drafts',exact:true}).click();
 connection=await recreated(oldCycle);page=connection.page;
 let operation=read('profile/maintenance.json');expect(operation.state).toBe('completed');expect(operation.action).toBe('backup');
 const backupId=operation.backupId;const backup=JSON.parse(fs.readFileSync(path.join(operation.backupDirectory,'desktop-backup.json')));
 expect(Object.keys(backup.manifest.files).some(n=>n.includes('dpapi')||n.includes('webview'))).toBe(false);
 expect(scientific()).toEqual(before);
 const captured=JSON.parse(fs.readFileSync(path.join(operation.backupDirectory,'desktop/session/checkpoint.json')));
 expect(captured.payload.entries[`chart:${project}`].value.title).toBe('D6 retained after maintenance');
 check('visible-editor-checkpoint-stop-full-backup-restart',{backupId,sourceFiles:Object.keys(backup.manifest.files).length});
 const selectBackup=async()=>{await page.getByRole('combobox',{name:'Desktop backup',exact:true}).click();await page.getByRole('option',{name:new RegExp(backupId.slice(0,8))}).click()};
 await selectBackup();let cycle=read('services.json').cycle;
 await page.getByRole('button',{name:'Verify backup',exact:true}).click();connection=await recreated(cycle);page=connection.page;
 expect(read('profile/maintenance.json').state).toBe('completed');expect(read('profile/maintenance.json').verified).toBe(true);expect(scientific()).toEqual(before);
 check('authenticated-ui-verification-reuses-complete-backup-contract');
 await selectBackup();cycle=read('services.json').cycle;
 await page.getByRole('button',{name:'Restore an independent copy',exact:true}).click();connection=await recreated(cycle);page=connection.page;
 operation=read('profile/maintenance.json');expect(operation.state).toBe('completed');expect(operation.action).toBe('restore');
 expect(scientific(operation.restoredProfileRoot)).toEqual(before);expect(scientific()).toEqual(before);
 const restored=JSON.parse(fs.readFileSync(path.join(operation.restoredProfileRoot,'session/checkpoint.json')));
 expect(restored.payload.profileId).toBe(operation.restoredProfileId);expect(restored.payload.profileId).not.toBe(captured.payload.profileId);
 expect(restored.payload.entries[`chart:${project}`].value.title).toBe('D6 retained after maintenance');
 check('independent-restore-keeps-scientific-identities-and-rebinds-desktop-session',{copyId:operation.copyId});
 await page.getByRole('button',{name:'Use restored copy on next launch',exact:true}).click();
 await expect.poll(()=>fs.existsSync(path.join(trial,'profile/profile-selection.json'))?read('profile/profile-selection.json').copyId:null).toBe(operation.copyId);
 await page.getByRole('button',{name:'Use original profile on next launch',exact:true}).click();
 await expect.poll(()=>fs.existsSync(path.join(trial,'profile/profile-selection.json'))).toBe(false);
 check('copy-and-original-selection-use-bounded-identifiers');
 const python=path.join(bundle,'app/runtime/python/python.exe');
 const sourceHealth=spawnSync(python,['-B','-X','utf8','-c',`import sys;from pathlib import Path;sys.path.insert(0,${JSON.stringify(path.join(bundle,'runtime'))});import installation;installation.health(Path(${JSON.stringify(bundle)}),Path(${JSON.stringify(operation.restoredProfileRoot)}));print('restored health ready')`],{windowsHide:true,encoding:'utf8',timeout:180000});
 fs.writeFileSync(path.join(trial,'restored-health.log'),sourceHealth.stdout+sourceHealth.stderr);expect(sourceHealth.status).toBe(0);
 check('independent-restored-profile-starts-bundled-authenticated-services');
 const packageId=JSON.parse(fs.readFileSync(path.join(bundle,'desktop-package.json'))).packageId;
 const prepared=spawnSync(python,['-B','-X','utf8',path.join(bundle,'runtime/installation.py'),'prepare','--install-root',path.join(trial,'maintenance-install'),'--profile-base',path.join(trial,'profile'),'--payload',path.join(bundle,'app'),'--package',packageId],{windowsHide:true,encoding:'utf8',timeout:250000});
 fs.writeFileSync(path.join(trial,'prepare-update.log'),prepared.stdout+prepared.stderr);expect(prepared.status).toBe(0);
 expect(fs.existsSync(path.join(trial,'profile/maintenance-request.json'))).toBe(false);
 operation=read('profile/maintenance.json');expect(operation.action).toBe('prepare-update');expect(operation.state).toBe('completed');expect(scientific()).toEqual(before);
 check('installer-preparation-checkpoints-and-gracefully-exits-live-desktop-services');
 report.packageId=packageId;report.passed=true;save();console.log(JSON.stringify({passed:true,checks:report.checks.length}));process.exit(0);
}catch(error){report.error=String(error.stack??error);save();console.error(report.error);process.exit(1)}
