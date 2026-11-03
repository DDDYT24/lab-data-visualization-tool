"""Hold a real multipart upload across the idle deadline in an owned headless session."""
import http.client
import importlib.util
import json
from pathlib import Path
import socket
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
trial=Path(sys.argv[1]).resolve()
assert trial.is_relative_to(Path.home()/'LabVizD2')
read=lambda name:json.loads((trial/name).read_text(encoding='utf-8-sig'))
spec=importlib.util.spec_from_file_location('upload_guard_controller',ROOT/'desktop/runtime/controller.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
spec=importlib.util.spec_from_file_location('upload_guard_runner',ROOT/'desktop/runtime/service_runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
services=read('profile/logs/desktop-services.json');session=read('session.json')
assert runner.port_owned(services['apiPort'],services['api']['pid'])
assert runner.record(runner.SimpleProcess(services['api']['pid']))==services['api']
key=controller.local_key(trial/'profile/appdata/LabVizV23/access.dpapi')
api=controller.ApiControl(services,key)
def command(action):
    target=trial/'commands.json';temporary=trial/'commands.new'
    temporary.write_text(json.dumps({'acceptanceId':session['id'],'id':uuid.uuid4().hex,'action':action}))
    temporary.replace(target)
def sleeps():
    host=read('profile/logs/desktop-host.json')
    return sum(json.loads(line)['event']=='services-suspended' for line in Path(host['hostLog']).read_text().splitlines())
before=sleeps()
boundary='LabVizAcceptance'+uuid.uuid4().hex
data=(ROOT/'outputs/desktop-d01/d0/fixtures/desktop-small.csv').read_bytes()
body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="held-upload.csv"\r\nContent-Type: text/csv\r\n\r\n').encode()+data+f'\r\n--{boundary}--\r\n'.encode()
with socket.create_connection(('127.0.0.1',services['apiPort']),timeout=60) as connection:
    headers=(f'POST /api/v1/projects HTTP/1.1\r\nHost: 127.0.0.1:{services["apiPort"]}\r\nCookie: labviz_local_session_v23_installed_{services["profileId"]}={key}\r\nContent-Type: multipart/form-data; boundary={boundary}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n').encode()
    connection.sendall(headers+body[:32])
    deadline=time.monotonic()+10
    while api.call('activity')['requests']==0 and time.monotonic()<deadline:time.sleep(.05)
    active=api.call('activity');assert not active['idle'] and active['requests']>=1
    command('hide');held=session['idleSeconds']+2;time.sleep(held)
    assert sleeps()==before,'An admitted upload was interrupted by suspension'
    assert api.call('activity')['requests']>=1
    connection.sendall(body[32:]);response=http.client.HTTPResponse(connection);response.begin();response.read()
    assert 200<=response.status<300,response.status
deadline=time.monotonic()+60
while not api.call('activity')['idle'] and time.monotonic()<deadline:time.sleep(.1)
assert api.call('activity')['idle']
command('show')
assert read('services.json')['api']['pid']==services['api']['pid']
result={'passed':True,'heldSeconds':held,'status':response.status,'requestsDuringUpload':active['requests'],'sameOwnedApi':True,'scope':'Real background multipart transfer; no native window actions'}
(trial/'upload-guard.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
