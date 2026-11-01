import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {EventEmitter} from 'node:events';
import {classifyCdpFailure, observeCdpConnection} from '../scripts/cdp_observer.mjs';

const base = {browserConnected:false, cycleAtAttach:1, currentCycle:1, hostEvents:[]};
test('a closed-target error without independent evidence stays unconfirmed', () => {
  assert.equal(classifyCdpFailure(base), 'unconfirmed');
});
test('an independent live-page observation distinguishes a client disconnect', () => {
  assert.equal(classifyCdpFailure({...base, probe:{reachable:true, pageAlive:true}}), 'client-transport-disconnected');
});
test('host crash, page closure, planned teardown and stale clients remain distinct', () => {
  for (const [event, expected, result] of [
    ['headless-page-crashed',false,'renderer-crashed'],
    ['headless-browser-disconnected',false,'host-browser-closed'],
    ['headless-page-closed',false,'application-page-closed'],
    ['headless-context-closed',true,'planned-host-teardown'],
  ]) assert.equal(classifyCdpFailure({...base, hostEvents:[{event, expected}]}), result);
  assert.equal(classifyCdpFailure({...base, currentCycle:2}), 'stale-client-after-rebuild');
});
test('telemetry is bounded and excludes URLs, credentials and raw error text', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'labviz-cdp-'));
  const browser = new EventEmitter();browser.isConnected = () => false;
  const page = new EventEmitter();page.isClosed = () => false;
  const observer = observeCdpConnection(browser, page, {directory,label:'test',cycleAtAttach:1,
    hostEventsPath:path.join(directory,'absent'),readCurrentCycle:() => 1,
    probe:async() => ({reachable:true,pageAlive:true,token:'private-probe-token'}),
    readOperation:() => ({action:'backup',state:'failed',requestId:'a'.repeat(32),message:'private-operation-message'})});
  try {
    for (let n=0;n<100;n++) browser.emit('disconnected');
    const value = await observer.capture(new Error('Target page, context or browser has been closed https://localhost/#labviz-access=private-key'));
    assert.equal(value.clientEvents.length,32);
    assert.equal(value.classification,'client-transport-disconnected');
    const text = fs.readFileSync(observer.path,'utf8');
    for (const secret of ['private-key','private-probe-token','private-operation-message','https://localhost']) assert.equal(text.includes(secret),false);
    assert.equal(value.operationAtCapture.state,'failed');
    assert.equal(value.errorKind,'target-closed');
    assert.equal(value.nativeAcceptance,false);
    observer.dispose();
    assert.equal(JSON.parse(fs.readFileSync(observer.path,'utf8')).classification,'client-transport-disconnected');
  } finally {
    observer.dispose();
    const resolved=fs.realpathSync(directory);
    assert.equal(path.dirname(resolved),fs.realpathSync(os.tmpdir()));
    assert.ok(path.basename(resolved).startsWith('labviz-cdp-'));
    fs.rmSync(resolved,{recursive:true,force:true});
  }
});
