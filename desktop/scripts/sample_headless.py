"""Sample owned application/test-browser trees without accessing desktop UI."""
import json
import sys
import time
from pathlib import Path
import psutil

trial=Path(sys.argv[1]);controller=int(sys.argv[2]);run_id=sys.argv[3];known={}
previous={}
def stop_record():
    try:
        value=json.loads((trial/'sampler-stop.json').read_text())
        return value if value.get('id')==run_id else None
    except (OSError,ValueError):return None
while not stop_record():
    members={}
    for pid in [controller, json.loads((trial/'browser-process.json').read_text()).get('pid') if (trial/'browser-process.json').exists() else None]:
        if not pid:continue
        try:
            owner=psutil.Process(pid)
            for p in [owner,*owner.children(recursive=True)]:members[p.pid]=p
        except psutil.NoSuchProcess:pass
    entries=[]
    for pid,p in members.items():
        try:
            memory=p.memory_info();known[pid]=p.create_time()
            now=time.monotonic();cpu=p.cpu_times();total=cpu.user+cpu.system
            key=(pid,p.create_time());last=previous.get(key)
            percent=100*(total-last[1])/(now-last[0]) if last and now>last[0] else None
            previous[key]=(now,total)
            entries.append({'pid':pid,'name':p.name(),'privateCommitBytes':memory.private,'workingSetBytes':memory.rss,'cpuPercentOneCore':percent})
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    with (trial/'resources.jsonl').open('a',encoding='utf-8') as stream:
        try:phase=json.loads((trial/'phase.json').read_text())['phase']
        except (OSError,ValueError,KeyError):phase='starting'
        stream.write(json.dumps({'utcSeconds':time.time(),'phase':phase,'privateCommitBytes':sum(p['privateCommitBytes'] for p in entries),'workingSetBytes':sum(p['workingSetBytes'] for p in entries),'cpuPercentOneCore':sum(p['cpuPercentOneCore'] or 0 for p in entries),'processes':entries,'scope':'Controller/services plus headless test browser; excludes test adapter and native host'})+'\n')
    time.sleep(.25)
time.sleep(2)
survivors=[]
for pid,created in known.items():
    try:
        p=psutil.Process(pid)
        if p.create_time()==created:survivors.append({'pid':pid,'name':p.name()})
    except psutil.NoSuchProcess:pass
(trial/'exit.json').write_text(json.dumps({'hostExitCode':stop_record()['hostExitCode'],'runId':run_id,'controllerPid':controller,'survivingOwnedProcesses':survivors,'scope':'Headless adapter exit; native host not launched'},indent=2))
