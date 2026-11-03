// Only the owned headless adapter is supported. This never connects to native apps.
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
const root=path.resolve(import.meta.dirname,'../..');
const {chromium,expect}=createRequire(path.join(root,'V2.0/web/package.json'))('@playwright/test');
const trial=path.resolve(process.argv[2]),bundle=path.resolve(process.argv[3]);
const read=name=>JSON.parse(fs.readFileSync(path.join(trial,name)));
const config=read('session.json');
if(config.shell!=='headless-test-adapter'||!trial.startsWith(path.join(process.env.USERPROFILE,'LabVizD2')+path.sep))throw Error('Use an owned headless trial');
const report={passed:false,packageId:JSON.parse(fs.readFileSync(path.join(bundle,'desktop-package.json'))).packageId,scope:'Headless production page and owned synthetic files only',nativeAcceptance:false,checks:[]};
const save=()=>fs.writeFileSync(path.join(trial,'part2-headless.json'),JSON.stringify(report,null,2));
const check=(name,details={})=>{report.checks.push({name,...details});save();console.log(name);};
let browser;
async function connect(){
  browser=await chromium.connectOverCDP(`http://127.0.0.1:${config.debugPort}`);
  const origin=read('services.json').webUrl;
  const page=browser.contexts()[0].pages().find(page=>page.url().startsWith(origin));
  if(!page)throw Error('Owned page unavailable');page.setDefaultTimeout(30000);return {page,origin};
}
try{
  let {page,origin}=await connect();
  const exports=read('export-links.json');
  expect(new Set(exports.map(item=>item.format))).toEqual(new Set(['png','svg','pdf','csv','json','zip']));
  const directory=path.join(trial,'中文资料', '科研路径'.repeat(30),'文件结果'.repeat(30));fs.mkdirSync(directory,{recursive:true});
  for(const extension of ['png','svg','pdf','csv','json','zip']){
    const item=exports.find(item=>item.format===extension);const response=await page.request.get(new URL(item.href,origin).toString());
    expect(response.ok()).toBe(true);const bytes=await response.body();expect(createHash('sha256').update(bytes).digest('hex')).toBe(item.sha256);
    const target=path.join(directory,'冻结结果.'+extension);expect(target.length).toBeGreaterThan(260);
    fs.writeFileSync(target,bytes);fs.writeFileSync(target,bytes);
    expect(createHash('sha256').update(fs.readFileSync(target)).digest('hex')).toBe(item.sha256);
    check('exact-frozen-format-and-long-unicode-file-'+extension,{bytes:bytes.length,pathLength:target.length,nativeSaveDialogTested:false});
  }
  const project=exports[0].href.match(/\/projects\/([^/]+)/)[1];
  await page.goto(origin+`/workspace/${project}?workspace=research&researchTask=analysis`);
  await page.getByRole('tab',{name:'Analysis',exact:true}).click();
  const closeDrawer=page.getByRole('button',{name:'Return to active task',exact:true});if(await closeDrawer.isVisible())await closeDrawer.click();
  const preview=page.getByRole('button',{name:'Preview PDF',exact:true});await preview.waitFor();await preview.focus();await page.keyboard.press('Enter');
  const frame=page.getByTitle('Saved PDF preview',{exact:true});await expect(frame).toBeVisible();
  const pdf=await page.request.get(new URL(await frame.getAttribute('src'),origin).toString());
  expect(pdf.headers()['content-type']).toContain('application/pdf');expect(pdf.headers()['content-disposition']).toContain('inline');expect((await pdf.body()).subarray(0,4).toString()).toBe('%PDF');
  await page.keyboard.press('Escape');await expect(frame).toHaveCount(0);await expect(preview).toBeFocused();
  check('inline-pdf-keyboard-open-escape-and-focus-return',{pdfRendererVisuallyAccepted:false});
  await page.getByRole('button',{name:'Save and computation details',exact:true}).click();
  await expect(page.getByTestId('desktop-data-tools')).toBeVisible();
  async function operation(button){
    const cycle=read('services.json').cycle;
    await page.getByRole('button',{name:button,exact:true}).click();
    await expect.poll(()=>read('services.json').cycle,{timeout:180000}).toBeGreaterThan(cycle);
    let connection;await expect.poll(async()=>{try{connection=await connect();return true}catch{return false}},{timeout:60000}).toBe(true);
    page=connection.page;origin=connection.origin;await expect(page.getByTestId('desktop-data-tools')).toBeVisible({timeout:60000});
    await expect.poll(()=>read('profile/maintenance.json').state,{timeout:30000}).toBe('completed');
    return read('profile/maintenance.json');
  }
  const backup=await operation('Back up data and drafts');
  async function chooseBackup(){
    await page.getByRole('combobox',{name:'Desktop backup',exact:true}).click();
    await page.getByRole('option').filter({hasText:backup.backupId.slice(0,8)}).click();
  }
  await chooseBackup();const first=await operation('Restore an independent copy');
  await chooseBackup();const second=await operation('Restore an independent copy');
  await chooseBackup();await operation('Verify backup');
  await expect.poll(async()=>page.getByRole('combobox',{name:'Restored profile',exact:true}).count()).toBe(1);
  await page.getByRole('combobox',{name:'Restored profile',exact:true}).click();
  await page.getByRole('option').filter({hasText:first.copyId.slice(0,8)}).click();
  await page.getByRole('button',{name:'Use selected copy on next launch',exact:true}).click();
  await expect(page.getByTestId('desktop-next-profile')).toContainText(first.copyId.slice(0,8));
  await expect(page.getByTestId('desktop-active-profile')).toContainText('Original profile');
  await page.getByRole('button',{name:'Use original profile on next launch',exact:true}).click();
  await expect(page.getByTestId('desktop-next-profile')).toContainText('Original profile');
  report.copies=[first,second].map(item=>({copyId:item.copyId,profileId:item.restoredProfileId,root:item.restoredProfileRoot}));
  check('backup-two-independent-restores-verify-and-select-older-copy',{copies:report.copies.length,activeProfileChangedWithoutRestart:false});
  await page.setViewportSize({width:640,height:450});
  const original=page.getByRole('button',{name:'Use original profile on next launch',exact:true});
  await original.scrollIntoViewIfNeeded();await original.focus();await page.keyboard.press('Enter');
  await expect(original).toBeEnabled();
  check('compact-viewport-data-controls-remain-keyboard-operable',{actualWindows200PercentZoomTested:false});
  report.passed=true;save();
}catch(error){report.error=String(error.stack??error);save();throw error;}
finally{if(browser)await browser.close();}
