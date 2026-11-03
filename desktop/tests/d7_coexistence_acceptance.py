"""Co-run copied V2.2/RC services and a live private desktop's services.

This verifies service/auth/data separation, not old launchers or native windows.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import ctypes
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

import psutil

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def call(url,key=None,cookie=None,post=False):
    request=urllib.request.Request(url,data=b'' if post else None,
        headers={**({'X-LabViz-Local-Key':key} if key else {}),**({'Cookie':cookie} if cookie else {})})
    try:
        with urllib.request.urlopen(request,timeout=5) as response:return response.status,response.read(),response.headers
    except urllib.error.HTTPError as error:return error.code,error.read(),error.headers


def port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));return listener.getsockname()[1]


def live(pid,created):
    try:
        if psutil.Process(pid).create_time()!=created:return False
    except psutil.NoSuchProcess:return False
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.WaitForSingleObject.restype=ctypes.c_ulong
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x100000,False,pid)
    if not handle:
        if ctypes.get_last_error()==87:return False
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        result=kernel.WaitForSingleObject(handle,0)
        if result not in {0,258}:raise ctypes.WinError(ctypes.get_last_error())
        return result==258  # A signaled terminated process can retain a PID while handles remain open.
    finally:kernel.CloseHandle(handle)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v22',type=Path,required=True);parser.add_argument('--v22-manifest',type=Path,required=True)
    parser.add_argument('--rc',type=Path,required=True);parser.add_argument('--current-profile',type=Path,required=True)
    parser.add_argument('--reuse-payloads',type=Path,help='Owned prior fixture; revalidate every payload byte before reuse')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    assert out.is_relative_to(ROOT/'outputs') and out!=ROOT/'outputs';out.mkdir(parents=True,exist_ok=False)
    trial=Path.home()/'LabVizD2'/('coexist-'+uuid.uuid4().hex[:10]);trial.mkdir()
    sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT/'V2.0/api')]
    from process_job import OwnedJob
    from labviz_api.installed_profile import InstalledProfile
    from controller import local_key
    from service_runner import port_owned
    from session import unlinked,io_path
    report={'format':'labviz-desktop-coexistence-v1','passed':False,'inProgress':True,'trial':str(trial),
            'scope':'Actual copied V2.2 release and historical V2.3 RC services alongside private desktop services. Old launchers, native entry points and default profiles are excluded.',
            'checks':[],'sources':{},'nativeAppLaunched':False}
    def save():(out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    jobs=[];children=[];logs=[];known={}
    def spawn(command,payload,environment,label):
        log=(out/(label+'.log')).open('wb');logs.append(log)
        child=subprocess.Popen(command,cwd=payload,env=environment,stdin=subprocess.DEVNULL,
            stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        children.append(child);known[child.pid]=psutil.Process(child.pid).create_time()
        if len(children)%2:jobs.append(OwnedJob(child))
        else:jobs[-1].add(child)
        return child
    original_local=os.environ.get('LOCALAPPDATA')
    try:
        instances=[]
        manifests={'v22':{x['path']:x['sha256'] for x in read(args.v22_manifest)['files']},
                   'rc':read(args.rc/'payload-manifest.json')['files']}
        for label,source in [('v22',args.v22.resolve()),('rc',args.rc.resolve())]:
            if args.reuse_payloads:
                prior=unlinked(args.reuse_payloads.absolute())
                assert prior.is_relative_to(Path.home()/'LabVizD2') and prior!=Path.home()/'LabVizD2'
                destination=unlinked(prior/label)
            else:
                destination=trial/label;destination.mkdir()
            members={name:digest for name,digest in manifests[label].items() if name.startswith(('runtime/','V2.0/'))}
            def copy(item):
                name,want=item;origin=unlinked(source/name);assert sha(origin)==want,(label,name)
                target=unlinked(destination/name);io_path(target.parent).mkdir(parents=True,exist_ok=True)
                if not args.reuse_payloads:shutil.copyfile(io_path(origin),io_path(target))
                assert sha(target)==want
            with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(copy,members.items()))
            report['sources'][label]={'root':str(source),'copiedVerifiedFiles':len(members),
                'manifestSha256':sha(args.v22_manifest if label=='v22' else source/'payload-manifest.json')}
            save();print(label+' verified payload copied',flush=True)
            appdata=trial/(label+'-appdata');appdata.mkdir()
            profile=appdata/('LabViz' if label=='v22' else 'LabVizV23')
            identifier=None
            if label=='rc':
                os.environ['LOCALAPPDATA']=str(appdata)
                identifier=InstalledProfile.open(profile,create=True).identifier
            data=profile/'data';data.mkdir(parents=True,exist_ok=True)
            api_port,web_port=port(),port()
            assert api_port!=web_port and not {api_port,web_port}&{3000,8000,3001,8001,3305,8305}
            key=secrets.token_urlsafe(32);origin=f'http://127.0.0.1:{web_port}'
            environment={k:v for k,v in os.environ.items() if not k.startswith('LABVIZ_') and k not in {'PYTHONPATH','PYTHONHOME','PYTHONPYCACHEPREFIX','NODE_PATH'}}
            environment.update(LOCALAPPDATA=str(appdata),LABVIZ_DATABASE_PATH=str(data/'labviz.db'),
                LABVIZ_OBJECT_STORAGE_ROOT=str(data/'objects'),LABVIZ_PERSISTENCE_BACKEND='sqlite',
                LABVIZ_OBJECT_STORAGE_BACKEND='local',LABVIZ_ENVIRONMENT='development',LABVIZ_AUTH_MODE='console',
                LABVIZ_LOCAL_ACCESS_KEY=key,LABVIZ_API_PROXY_TARGET=f'http://127.0.0.1:{api_port}',
                LABVIZ_ALLOWED_ORIGINS=origin,LABVIZ_PUBLIC_WEB_URL=origin,PORT=str(web_port),HOSTNAME='127.0.0.1',
                NEXT_TELEMETRY_DISABLED='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(profile/'cache/matplotlib'),
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                HTTP_PROXY='http://127.0.0.1:9',HTTPS_PROXY='http://127.0.0.1:9',NO_PROXY='127.0.0.1,localhost,::1')
            if label=='rc':environment.update(LABVIZ_INSTALL_LINE='2.3',LABVIZ_DEVELOPMENT_LINE='2.3',LABVIZ_PROFILE_ROOT=str(profile),LABVIZ_PROFILE_ID=identifier,LABVIZ_INSTALL_ROOT=str(destination))
            command=[str(destination/'runtime/python/python.exe'),'-B','-X','utf8','-m']
            command+=['labviz_api.installed_server','--port',str(api_port)] if label=='rc' else ['uvicorn','labviz_api.main:app','--host','127.0.0.1','--port',str(api_port),'--no-access-log']
            api=spawn(command,destination/'V2.0/api',environment,label+'-api')
            web=spawn([str(destination/'runtime/node/node.exe'),str(destination/'V2.0/web/server.js')],destination,environment,label+'-web')
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                assert api.poll() is None and web.poll() is None,label+' service exited'
                try:
                    status,_,headers=call(f'http://127.0.0.1:{api_port}/api/v1/local/session',key,post=True)
                    if status==200 and call(origin)[0]==200:break
                except OSError:pass
                time.sleep(.1)
            else:raise TimeoutError(label+' readiness')
            assert port_owned(api_port,api.pid) and port_owned(web_port,web.pid)
            cookie=headers.get('Set-Cookie').split(';',1)[0]
            instances.append({'label':label,'api':f'http://127.0.0.1:{api_port}',
                              'web':origin,'key':key,'cookie':cookie,'profile':str(profile)})
            print(label+' actual API and web ready',flush=True)
        current=read(args.current_profile/'logs/desktop-services.json')
        current_key=local_key(args.current_profile/'appdata/LabVizV23/access.dpapi')
        instances.append({'label':'desktop','api':f"http://127.0.0.1:{current['apiPort']}",
            'web':current['webUrl'],'key':current_key,
            'cookie':f"labviz_local_session_v23_installed_{current['profileId']}={current_key}",
            'profile':str(args.current_profile/'appdata/LabVizV23')})
        assert len({instance['profile'].casefold() for instance in instances})==3
        assert len({instance['api'] for instance in instances}|{instance['web'] for instance in instances})==6
        assert len({instance['cookie'].split('=',1)[0] for instance in instances})==3
        for target in instances:
            assert call(target['api']+'/api/v1/projects',cookie=target['cookie'])[0]==200
            assert call(target['web']+'/api/v1/projects',cookie=target['cookie'])[0]==200
            for foreign in instances:
                if foreign is target:continue
                assert call(target['api']+'/api/v1/projects',cookie=foreign['cookie'])[0]==403
                assert call(target['api']+'/api/v1/local/session',foreign['key'],post=True)[0]==403
            report['checks'].append({'name':target['label']+'-separate-live-services-and-cross-cookie-key-refusal',
                'ownApiAndProxyAuthenticated':True,'foreignCookiesRefused':2,'foreignKeysRefused':2,
                'cookieName':target['cookie'].split('=',1)[0],'profile':target['profile']})
        report.update(passed=True,inProgress=False)
    except BaseException as error:report.update(inProgress=False,error=repr(error));raise
    finally:
        for child in children:
            try:
                for member in psutil.Process(child.pid).children(recursive=True):known[member.pid]=member.create_time()
            except psutil.NoSuchProcess:pass
        report['ownedProcessIdentities']=[{'pid':pid,'created':created} for pid,created in known.items()]
        for job in jobs:job.close()
        for child in children:
            if child.poll() is None:child.terminate()
            child.wait(timeout=15)
        for log in logs:log.close()
        deadline=time.monotonic()+5
        while True:
            survivors=[pid for pid,created in known.items() if live(pid,created)]
            if not survivors or time.monotonic()>deadline:break
            time.sleep(.1)
        report['childExitCodes']=[child.returncode for child in children]
        report['ownedSurvivors']=survivors
        if survivors:report['passed']=False
        if original_local is None:os.environ.pop('LOCALAPPDATA',None)
        else:os.environ['LOCALAPPDATA']=original_local
        save()
    assert report['passed'],report
    print(json.dumps({'passed':True,'checks':len(report['checks']),'ownedSurvivors':survivors}))


if __name__=='__main__':main()
