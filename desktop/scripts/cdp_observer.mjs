import fs from 'node:fs';
import path from 'node:path';
import {randomUUID} from 'node:crypto';

const kinds = new Set(['headless-page-closed', 'headless-page-crashed', 'headless-context-closed', 'headless-browser-disconnected']);

export function classifyCdpFailure(value) {
  const events = value.hostEvents ?? [];
  if (events.some(event => event.event === 'headless-page-crashed') || value.clientEvents?.some(event => event.event === 'client-page-crashed')) return 'renderer-crashed';
  if (events.some(event => event.event === 'headless-browser-disconnected' && event.expected === false)) return 'host-browser-closed';
  if (events.some(event => event.event === 'headless-page-closed' && event.expected === false)) return 'application-page-closed';
  if (events.some(event => kinds.has(event.event) && event.expected === true)) return 'planned-host-teardown';
  if (Number.isInteger(value.currentCycle) && value.currentCycle > value.cycleAtAttach) return 'stale-client-after-rebuild';
  if (!value.browserConnected && value.probe?.reachable === true && value.probe.pageAlive === true) return 'client-transport-disconnected';
  if (value.errorKind === null && value.browserConnected && !value.pageClosed) return 'healthy-connection';
  return 'unconfirmed';
}

function errorKind(error) {
  const text = String(error?.message ?? '');
  if (/Target page, context or browser has been closed|TargetClosedError/i.test(text)) return 'target-closed';
  if (/timeout|timed out/i.test(text)) return 'timeout';
  if (/connection|websocket|protocol error/i.test(text)) return 'transport';
  return error ? 'other' : null;
}

export function observeCdpConnection(browser, page, options) {
  if (!/^[a-z0-9-]{1,48}$/i.test(options.label)) throw new Error('Invalid diagnostic label');
  const directory = path.resolve(options.directory);
  for (let member = directory; ; member = path.dirname(member)) {
    if (fs.existsSync(member) && fs.lstatSync(member).isSymbolicLink()) throw new Error('Linked diagnostic output');
    if (path.dirname(member) === member) break;
  }
  fs.mkdirSync(directory, {recursive:true});
  const identifier = randomUUID();
  const target = path.join(directory, options.label + '-' + identifier + '.json');
  const attachedUtcSeconds = Date.now()/1000;
  const cycleAtAttach = options.cycleAtAttach;
  const events = [];
  let disposed = false;
  let lastError = null, lastProbe;

  function hostEvents() {
    try {
      return fs.readFileSync(options.hostEventsPath, 'utf8').split('\n').filter(Boolean).map(JSON.parse)
        .filter(event => kinds.has(event.event) && event.utcSeconds >= attachedUtcSeconds && event.cycle === cycleAtAttach)
        .slice(-32).map(event => ({event:event.event, utcSeconds:event.utcSeconds, cycle:event.cycle, expected:event.expected}));
    } catch { return []; }
  }
  function snapshot(error = lastError, probe = lastProbe) {
    let currentCycle = null;
    try { const value = options.readCurrentCycle(); if (Number.isInteger(value)) currentCycle = value; } catch { /* unavailable during teardown */ }
    const value = {
      format:'labviz-cdp-observation-v1', identifier, label:options.label,
      attachedUtcSeconds, observedUtcSeconds:Date.now()/1000, cycleAtAttach, currentCycle,
      browserConnected:browser.isConnected(), pageClosed:page.isClosed(),
      errorKind:errorKind(error), clientEvents:[...events], hostEvents:hostEvents(),
      nativeAcceptance:false,
      scope:'Test-client metadata only; no URL, headers, request body or error text retained',
    };
    if (probe) value.probe = {reachable:probe.reachable === true, pageAlive:probe.pageAlive === true};
    if (options.readOperation) {
      try {
        const operation = options.readOperation();
        value.operationAtCapture = {
          action:['backup','verify','restore','clear-cache','prepare-update'].includes(operation.action) ? operation.action : null,
          state:['waiting','running','completed','failed'].includes(operation.state) ? operation.state : null,
          requestId:/^[a-f0-9]{32}$/i.test(operation.requestId ?? '') ? operation.requestId : null,
        };
      } catch { /* no operation has been admitted */ }
    }
    value.classification = classifyCdpFailure(value);
    return value;
  }
  function persist(value) {
    const temporary = target + '.new-' + randomUUID();
    fs.writeFileSync(temporary, JSON.stringify(value, null, 2));
    fs.renameSync(temporary, target);
    return value;
  }
  function record(event) {
    if (disposed) return;
    events.push({event, utcSeconds:Date.now()/1000});
    if (events.length > 32) events.shift();
    persist(snapshot());
  }
  const disconnected = () => record('cdp-client-disconnected');
  const closed = () => record('client-page-closed');
  const crashed = () => record('client-page-crashed');
  browser.on('disconnected', disconnected);page.on('close', closed);page.on('crash', crashed);
  persist(snapshot(null));
  return {
    path:target,
    async capture(error) {
      lastError = error;
      if (options.probe) { try { lastProbe = await options.probe(); } catch { lastProbe = {reachable:false, pageAlive:false}; } }
      return persist(snapshot());
    },
    dispose() {
      if (disposed) return;
      persist(snapshot());disposed = true;
      browser.off('disconnected', disconnected);page.off('close', closed);page.off('crash', crashed);
    },
  };
}
