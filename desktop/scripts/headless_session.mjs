// Background-only integration host. It does not launch the native desktop executable.
import fs from 'node:fs';
import path from 'node:path';
import net from 'node:net';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { createRequire } from 'node:module';
import { randomUUID, createHash } from 'node:crypto';

const root=path.resolve(import.meta.dirname,'../..');
const require=createRequire(path.join(root,'V2.0/web/package.json'));
const {chromium}=require('@playwright/test');
const label=process.argv[2];
if(!/^[a-zA-Z0-9-]+$/.test(label))throw new Error('Invalid acceptance label');
const trial=path.join(process.env.USERPROFILE,'LabVizD2',label);
const restart=process.argv.includes('--restart');
if(restart){
  const previousHost=JSON.parse(fs.readFileSync(path.join(trial,'host.json')));
  try{process.kill(previousHost.pid,0);throw new Error('The previous headless host is still running');}catch(error){if(error.code!=='ESRCH')throw error;}
  const previous=JSON.parse(fs.readFileSync(path.join(trial,'exit.json')));
  if(previous.hostExitCode!==0||previous.survivingOwnedProcesses.length)throw new Error('The previous owned session did not exit cleanly');
  const archive=['session.json','host.json','services.json','headless-events.jsonl','exit.json','sampler-stop.json','browser-process.json',...fs.readdirSync(trial).filter(name=>/^(?:workflow-|suspend-|probe-|page-errors|resources|export-links|content-verification|upload-guard|failure-|webgl-surface)/.test(name))];
  for(const name of new Set(archive)){
    const old=path.join(trial,name);if(fs.existsSync(old))fs.renameSync(old,path.join(trial,Date.now()+'-'+name));
  }
}else fs.mkdirSync(trial,{recursive:false});
const bundle=path.resolve(process.argv[3]??path.join(root,'outputs/desktop-d34/bundle'));
const profile=path.join(trial,'profile');
const manifest=JSON.parse(fs.readFileSync(path.join(bundle,'desktop-runtime.json')));
for(const [name,key] of [['controller.py','controllerSha256'],['preferences.py','preferencesSha256'],['process_job.py','processJobSha256'],['session.py','sessionSha256'],['service_runner.py','serviceRunnerSha256']]) {
  if(createHash('sha256').update(fs.readFileSync(path.join(bundle,'runtime',name))).digest('hex')!==manifest[key])throw new Error('Runtime hash differs: '+name);
}
const port=await new Promise(resolve=>{const server=net.createServer();server.listen(0,'127.0.0.1',()=>{const port=server.address().port;server.close(()=>resolve(port));});});
const write=(name,value)=>{const target=path.join(trial,name);fs.writeFileSync(target+'.new',JSON.stringify(value,null,2));fs.renameSync(target+'.new',target);};
const id=randomUUID().replaceAll('-','');
write('session.json',{format:'labviz-desktop-acceptance-v1',id,profileRoot:profile,debugPort:port,idleSeconds:Number(process.argv[4]??10),shell:'headless-test-adapter'});
write('launcher.json',JSON.parse(fs.readFileSync(path.join(trial,'session.json'))));
write('controller-session.json',{format:'labviz-desktop-session-v1',profileRoot:profile,payloadRoot:path.join(bundle,'app'),candidateId:manifest.candidateId,startupTimeoutSeconds:180,idleSeconds:Number(process.argv[4]??10)});
write('host.json',{pid:process.pid,startedUtcSeconds:Date.now()/1000,scope:'Headless test adapter; not native-window or tray evidence'});
const python=path.join(bundle,'app/runtime/python/python.exe');
const error=fs.openSync(path.join(trial,'controller-errors.log'),'a');
const controller=spawn(python,['-B','-X','utf8',path.join(bundle,'runtime/controller.py'),path.join(trial,'controller-session.json')],{windowsHide:true,stdio:['pipe','pipe',error]});
let browser,page,browserLifetime,visible=true,committing=false,deferredShow=false,cycle=0,exiting=false,failed=false;
const send=action=>controller.stdin.write(JSON.stringify({action})+'\n');
const eventLog=path.join(trial,'headless-events.jsonl');
const event=(name,details={})=>fs.appendFileSync(eventLog,JSON.stringify({event:name,utcSeconds:Date.now()/1000,scope:'headless-test-adapter',...details})+'\n');
controller.on('exit',(code,signal)=>event('controller-process-exited',{code,signal,expected:exiting||failed}));
let previous='';
const timer=setInterval(()=>{
  try {
    const command=JSON.parse(fs.readFileSync(path.join(trial,'commands.json')));
    if(command.acceptanceId!==id||!command.id||command.id===previous)return;
    previous=command.id;
    if(command.action==='show'){visible=true;if(committing)deferredShow=true;else send('show');event('client-shown');}
    if(command.action==='hide'){visible=false;send('hide');event('client-hidden');}
    if(command.action==='exit'){exiting=true;send('stop');}
    if(['exit-wait','exit-cancel','exit-confirm'].includes(command.action))send(command.action);
    if(command.action.startsWith('download-start:')||command.action.startsWith('download-end:'))send(command.action);
  }catch{}
},100);
write('controller-process.json',{pid:controller.pid});
const sampler=spawn(process.env.LABVIZ_TEST_PYTHON??'python',['-B',path.join(root,'desktop/scripts/sample_headless.py'),trial,String(controller.pid),id],{windowsHide:true,stdio:'ignore'});
console.log(JSON.stringify({trial,controllerPid:controller.pid,debugPort:port,scope:'No desktop controls; headless integration only'}));
try {
  for await(const line of createInterface({input:controller.stdout})) {
    const message=JSON.parse(line);
    if(message.type==='ready') {
      cycle++;
      const context=await chromium.launchPersistentContext(path.join(trial,'browser-'+cycle),{headless:true,viewport:{width:1440,height:1000},args:[`--remote-debugging-port=${port}`,'--remote-debugging-address=127.0.0.1','--enable-unsafe-swiftshader']});
      browser=context.browser();
      const lifetime={cycle,closing:false};browserLifetime=lifetime;
      const closure=name=>event(name,{cycle:lifetime.cycle,expected:lifetime.closing||exiting});
      browser.on('disconnected',()=>closure('headless-browser-disconnected'));
      context.on('close',()=>closure('headless-context-closed'));
      const system=await browser.newBrowserCDPSession();const processes=await system.send('SystemInfo.getProcessInfo');
      write('browser-process.json',{pid:processes.processInfo.find(p=>p.type==='browser').id});await system.detach();
      const origin=new URL(message.url).origin;
      const adapter=fs.readFileSync(path.join(root,'desktop/runtime/adapter.js'),'utf8').replace('__LABVIZ_DESKTOP_CONFIGURATION__',JSON.stringify({origin,profileId:message.profileId,preferences:message.preferences}));
      await context.addInitScript(adapter);
      page=context.pages()[0]??await context.newPage();
      page.on('close',()=>closure('headless-page-closed'));
      page.on('crash',()=>closure('headless-page-crashed'));
      page.on('pageerror',error=>fs.appendFileSync(path.join(trial,'page-errors.jsonl'),JSON.stringify({message:error.message,cycle})+'\n'));
      await page.goto(message.url,{waitUntil:'domcontentloaded',timeout:60000});
      await page.waitForFunction(()=>!location.hash.startsWith('#labviz-access='),{timeout:60000});
      write('services.json',{...JSON.parse(fs.readFileSync(path.join(profile,'logs/desktop-services.json'))),cycle});
      write('profile/logs/desktop-host.json',{pid:process.pid,profileId:message.profileId,origin,hostLog:eventLog,automaticSuspension:true,scope:'Headless adapter; native window not tested'});
      event('headless-page-created');console.log(JSON.stringify({ready:true,cycle,origin}));
    }else if(message.type==='checkpoint') {
      event('checkpoint-requested');
      if(visible&&!message.maintenance){send('show');continue;}
      // Do not block control processing while a deliberately held checkpoint is tested.
      void page.evaluate(id=>window.labvizDesktopCheckpoint?.(id),message.checkpointId).catch(error=>event('checkpoint-page-unavailable'));
    }else if(message.type==='suspend-commit') {
      if(visible&&!message.maintenance){send('show');continue;}
      committing=true;browserLifetime.closing=true;await browser.close();browser=null;page=null;
      write('browser-process.json',{pid:null});event('headless-page-destroyed');send('webview-destroyed');
      committing=false;if(deferredShow){deferredShow=false;send('show');}
    }else if(message.type==='suspended'){event('services-suspended');console.log(JSON.stringify({suspended:true,cycle}));}
    else if(message.type==='sleep-cancelled')event('suspension-cancelled');
    else if(message.type==='sleep-blocked'){event('suspension-blocked');console.log(JSON.stringify({suspensionBlocked:true,message:message.message}));}
    else if(message.type==='maintenance-exit'){event('maintenance-exit');exiting=true;}
    else if(['exit-ready','exit-waiting','exit-blocked','exit-cancelled'].includes(message.type)){event(message.type);}
    else if(message.type==='exit-approved'){event('exit-approved');exiting=true;}
    else if(message.type==='failed'){event('controller-failed');throw new Error(message.message);}
  }
}catch(error){console.error(String(error.stack??error));failed=true;exiting=true;send('stop');}
finally {
  clearInterval(timer);if(browser){browserLifetime.closing=true;await browser.close();}write('browser-process.json',{pid:null});
  await new Promise(resolve=>controller.exitCode!==null?resolve():controller.once('exit',resolve));
  write('sampler-stop.json',{stop:true,id,hostExitCode:failed?1:(controller.exitCode??1)});
  await new Promise(resolve=>sampler.exitCode!==null?resolve():sampler.once('exit',resolve));
  console.log(JSON.stringify({completed:true,exiting,controllerExitCode:controller.exitCode}));
  process.exit(failed?1:0);
}
