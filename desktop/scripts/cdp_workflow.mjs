import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { createHash, randomUUID } from 'node:crypto';

const root = path.resolve(import.meta.dirname, '../..');
const require = createRequire(path.join(root, 'V2.0/web/package.json'));
const { chromium, expect } = require('@playwright/test');
const trial = path.resolve(process.argv[2]);
const mode = process.argv[3] ?? 'full';
const config = JSON.parse(fs.readFileSync(path.join(trial, 'launcher.json')));
const output = path.join(trial, `artifacts-${mode}`);
fs.mkdirSync(output, { recursive: true });
const report = { shell: config.shell, mode, passed: false, checks: [], timings: {}, exports: [], pageErrors: [], externalRequests: [] };
let page;
function save() { fs.writeFileSync(path.join(trial, `workflow-${mode}.json`), JSON.stringify(report, null, 2)); }
function check(name, details = {}) { report.checks.push({ name, ...details }); save(); console.log(name); }
function phase(name) { fs.writeFileSync(path.join(trial, 'phase.json'), JSON.stringify({ phase: name })); }
function command(action) {
  const desktop = config.format === 'labviz-desktop-acceptance-v1';
  fs.writeFileSync(path.join(trial, desktop ? 'commands.json' : 'host-command.json'), JSON.stringify({ ...(desktop ? { acceptanceId: config.id } : { trialId: config.trialId }), id: randomUUID(), action }));
}
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

try {
  const deadline = Date.now() + 180000;
  let services;
  while (Date.now() < deadline) {
    if (fs.existsSync(path.join(trial, 'services.json'))) { services = JSON.parse(fs.readFileSync(path.join(trial, 'services.json'))); break; }
    await pause(150);
  }
  if (!services) throw new Error('No owned production service readiness record');
  const origin = services.webUrl;
  let browser;
  while (Date.now() < deadline) {
    try { browser = await chromium.connectOverCDP(`http://127.0.0.1:${config.debugPort}`); break; } catch { await pause(200); }
  }
  if (!browser) throw new Error('Native WebView2 CDP endpoint did not become available');
  const context = browser.contexts()[0];
  await expect.poll(() => context.pages().filter(p => p.url().startsWith(origin)).length, { timeout: 30000 }).toBeGreaterThan(0);
  page = context.pages().find(p => p.url().startsWith(origin));
  page.setDefaultTimeout(30000);
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext;
    window.__labvizD1WebGL = [];
    HTMLCanvasElement.prototype.getContext = function (...args) {
      const result = Reflect.apply(original, this, args);
      if (result && ['webgl', 'webgl2', 'experimental-webgl'].includes(args[0])) {
        window.__labvizD1WebGL.push({ canvas: this, context: result, kind: args[0] });
      }
      return result;
    };
  });
  page.on('pageerror', error => report.pageErrors.push(error.message));
  page.on('request', request => { const url = new URL(request.url()); if (url.protocol.startsWith('http') && !['127.0.0.1', 'localhost'].includes(url.hostname)) report.externalRequests.push(url.origin + url.pathname); });
  const url = relative => new URL(relative, origin).toString();
  const get = relative => page.request.get(url(relative), { timeout: 180000 });
  const post = (relative, options) => page.request.post(url(relative), { timeout: 180000, ...options });
  const api = async response => { expect(response.ok(), (await response.text()).slice(0, 500)).toBeTruthy(); return response.json(); };
  await page.goto(origin + '/');
  await page.locator('input[type="file"]').first().waitFor({ state: 'attached' });
  report.timings.observedHomeReadyMs = Date.now() - JSON.parse(fs.readFileSync(path.join(trial, 'host.json'))).startedUtcSeconds * 1000;
  report.timings.servicesReadyMs = services.serviceReadyMs;
  phase('home-idle');
  await pause(3000);
  check(config.shell==='headless-test-adapter'?'headless-production-home-and-local-session':'native-webview-home-and-local-session');
  if (mode === 'smoke') {
    command('hide'); phase('hidden-idle'); await pause(5000);
    command('show'); phase('restored-idle'); await pause(2000);
    expect((await get('/api/v1/projects')).ok()).toBeTruthy();
    check('hide-restore-preserves-local-session');
  } else {
    await post('/api/locale', { data: { locale: 'en' } });
    await page.reload();
    const initialPreferences = await page.evaluate(() => ({appearance:JSON.parse(localStorage.getItem('labviz:user-preferences:v1') ?? '{}').appearance, mode:localStorage.getItem('labviz:v23:interface-mode:v1')}));
    const lightSwitch = page.getByRole('button', { name: 'Switch to light theme', exact: true });
    if (initialPreferences.appearance === 'dark') { await expect(lightSwitch).toBeVisible(); await lightSwitch.click(); }
    else if (await lightSwitch.isVisible()) await lightSwitch.click();
    const classicSwitch = page.getByRole('button', { name: 'Switch to classic interface', exact: true });
    if (initialPreferences.mode === 'overview') { await expect(classicSwitch).toBeVisible(); await classicSwitch.click(); }
    else if (await classicSwitch.isVisible()) await classicSwitch.click();
    const fixtures = path.join(root, 'outputs/desktop-d01/d0/fixtures');
    let projectId;
    for (const file of mode === 'large' ? ['desktop-100k.csv'] : ['desktop-small.tsv', 'desktop-small.xlsx', path.join('中文 路径', '科研数据.csv')]) {
      phase('import');
      await page.goto(origin + '/');
      const started = Date.now();
      await page.locator('input[type="file"]').setInputFiles(path.join(fixtures, file));
      await expect(page.getByRole('heading', { name: 'Confirm your data', exact: true })).toBeVisible({ timeout: 120000 });
      projectId = new URL(page.url()).pathname.split('/').at(-1);
      const workspace = await api(await get(`/api/v1/projects/${projectId}/workspace`));
      expect(workspace.preview.totalRows).toBe(mode === 'large' ? 100000 : 12);
      check('real-file-import', { file, rows: workspace.preview.totalRows, durationMs: Date.now() - started });
    }
    const base = `/api/v1/projects/${projectId}/research`;
    await page.getByRole('button', { name: 'Research workspace', exact: true }).click();
    const research = page.getByTestId('research-workspace');
    await expect(research.getByText('Saved state', { exact: true })).toBeVisible();
    const researchFile = path.join(fixtures, mode === 'large' ? 'desktop-100k.csv' : 'desktop-small.csv');
    await research.locator('input[type="file"]').first().setInputFiles(researchFile);
    await research.getByRole('button', { name: 'More metadata, limits and experimental records', exact: true }).click();
    for (const [label, field] of [['Measurement value', 'value'], ['Sample ID', 'sample'], ['Independent experimental unit', 'sample'], ['Unit column', 'unit'], ['Group', 'group'], ['Time point', 'time'], ['Pair key', 'pair']]) {
      await research.getByRole('combobox', { name: label, exact: true }).fill(field);
      await research.getByRole('combobox', { name: label, exact: true }).press('Tab');
    }
    phase('research-preparation');
    await research.getByRole('button', { name: 'Preview preparation', exact: true }).click();
    await expect(research.getByTestId('research-counts')).toContainText(`Observations: ${mode === 'large' ? 100000 : 12}`, { timeout: 120000 });
    report.preparationCounts = await research.getByTestId('research-counts').innerText();
    await research.getByRole('button', { name: 'Apply preview and save revision', exact: true }).click();
    await page.getByRole('tab', { name: 'Analysis', exact: true }).click();
    const charts = page.getByTestId('research-charts');
    await expect(charts.getByRole('button', { name: 'Generate and save chart', exact: true })).toBeEnabled({ timeout: mode === 'large' ? 120000 : 30000 });
    check('research-preparation-and-save', { counts: report.preparationCounts });

    async function artifact(href, extension, label) {
      const response = await get(href);
      expect(response.ok()).toBeTruthy();
      const bytes = await response.body();
      expect(bytes.length).toBeGreaterThan(extension === 'json' || extension === 'csv' ? 10 : 1000);
      if (extension === 'png') expect(bytes.subarray(1, 4).toString()).toBe('PNG');
      if (extension === 'pdf') expect(bytes.subarray(0, 4).toString()).toBe('%PDF');
      if (extension === 'svg') expect(bytes.toString()).toContain('<svg');
      if (extension === 'zip') expect(bytes.subarray(0, 2).toString()).toBe('PK');
      const target = path.join(output, `${label}.${extension}`);
      fs.writeFileSync(target, bytes);
      const record = { label, format: extension, href, bytes: bytes.length, sha256: hash(bytes), file: target };
      report.exports.push(record); save();
      return bytes;
    }

    const families = mode === 'large' ? [['Grouped raw points', 'D1 Large raw']] : [['Grouped raw points', 'D1 Raw'], ['Paired change', 'D1 Paired'], ['Violin distribution', 'D1 Violin']];
    for (const [family, title] of families) {
      phase('chart-generation');
      await charts.getByRole('combobox', { name: 'Chart family', exact: true }).click();
      await page.getByRole('option', { name: family, exact: true }).click();
      await charts.getByRole('textbox', { name: 'Chart title', exact: true }).fill(title);
      const started = Date.now();
      await charts.getByRole('button', { name: 'Generate and save chart', exact: true }).click();
      const image = charts.getByRole('img', { name: title, exact: true });
      await expect.poll(() => image.evaluate(el => el.naturalWidth), { timeout: 180000 }).toBeGreaterThan(0);
      const jsonHref = await charts.getByRole('link', { name: 'Download JSON', exact: true }).getAttribute('href');
      const frozen = JSON.parse((await artifact(jsonHref, 'json', title.replaceAll(' ', '-'))).toString());
      expect(frozen.result.points).toHaveLength(mode === 'large' ? 100000 : 12);
      if (family === 'Paired change') expect(frozen.result.pairs.map(pair => pair.delta)).toEqual([2, 2, 2, 2, 2, 2]);
      if (family === 'Violin distribution') expect(frozen.result.densities[0].grid.length).toBeGreaterThanOrEqual(129);
      for (const format of ['PNG', 'SVG', 'PDF']) await artifact(await charts.getByRole('link', { name: `Download ${format}`, exact: true }).getAttribute('href'), format.toLowerCase(), title.replaceAll(' ', '-') );
      check('scientific-chart-and-frozen-content', { family, durationMs: Date.now() - started });
    }
    phase('research-idle'); await pause(3000);
    await page.getByRole('tab', { name: 'Figures', exact: true }).click();
    const board = page.getByTestId('figure-workspace');
    const select = async (region, label, option) => { await region.getByRole('combobox', { name: label, exact: true }).click(); await page.getByRole('option', { name: option, exact: typeof option === 'string' }).click(); };
    await select(board, 'Chart for panel A', mode === 'large' ? /D1 Large raw/ : /D1 Raw/);
    await select(board, 'Panel count', mode === 'large' ? '4' : '2');
    for (const letter of mode === 'large' ? ['B', 'C', 'D'] : ['B']) await select(board, `Chart for panel ${letter}`, mode === 'large' ? /D1 Large raw/ : /D1 Violin/);
    const title = mode === 'large' ? 'D1 Large four panels' : 'D1 Two panels';
    await board.getByRole('textbox', { name: 'Figure title', exact: true }).fill(title);
    const committed = page.waitForResponse(response => response.request().method() === 'POST' && response.url().endsWith('/research/figure-boards?deferArtifacts=true'));
    phase('figure-generation'); const generated = Date.now();
    await board.getByRole('button', { name: 'Save new layout revision', exact: true }).click();
    const response = await committed;
    expect(response.status()).toBe(202);
    const queued = await response.json();
    command('hide');
    await expect.poll(async () => (await api(await get(base + '/figure-boards/' + queued.id))).composed, { timeout: 180000 }).toBe(true);
    command('show');
    await expect.poll(() => board.getByRole('img', { name: 'Saved composed figure', exact: true }).evaluate(el => el.naturalWidth), { timeout: 60000 }).toBeGreaterThan(0);
    for (const format of ['PNG', 'SVG', 'PDF']) await artifact(await board.getByRole('link', { name: `Download figure ${format}`, exact: true }).getAttribute('href'), format.toLowerCase(), 'D1-composed');
    check('composed-generation-continues-while-hidden', { panels: mode === 'large' ? 4 : 2, durationMs: Date.now() - generated });
    await board.getByRole('textbox', { name: 'Figure title', exact: true }).fill('D1 retained unsaved draft');
    command('hide'); phase('hidden-idle'); await pause(5000); command('show'); phase('restored-idle'); await pause(2000);
    await expect(board.getByRole('textbox', { name: 'Figure title', exact: true })).toHaveValue('D1 retained unsaved draft');
    check('hide-show-retains-unsaved-draft-without-destroying-window');
    const reopened = Date.now();
    await page.reload();
    await expect(board.getByRole('textbox', { name: 'Figure title', exact: true })).toHaveValue(title);
    report.timings.savedProjectReloadMs = Date.now() - reopened;
    check('saved-project-reopen');
    if (mode === 'full') {
      await page.getByRole('tab', { name: 'Outputs', exact: true }).click();
      const reuse = page.getByTestId('reuse-workspace');
      await select(reuse, 'Saved figure for template / output', /D1 Two panels/);
      await reuse.getByRole('textbox', { name: 'Template name', exact: true }).fill('D1 workflow template');
      await reuse.getByRole('button', { name: 'Save workflow template', exact: true }).click();
      await expect(reuse.getByRole('status')).toHaveText('Template saved locally.');
      const replacement = path.join(trial, 'replacement.csv');
      fs.writeFileSync(replacement, fs.readFileSync(researchFile, 'utf8').replace('sample,value,', 'sample,voltage,').replace('s6,8,', 's6,18,'));
      await reuse.getByTestId('replacement-file').setInputFiles(replacement);
      await select(reuse, 'Map value', 'voltage');
      await reuse.getByRole('button', { name: 'Validate replacement', exact: true }).click();
      await expect(reuse.getByTestId('replacement-preview')).toContainText('A: ready');
      await reuse.getByRole('button', { name: 'Apply validated replacement', exact: true }).click();
      await expect(reuse.getByRole('status')).toHaveText('Replacement saved as new revisions.', { timeout: 180000 });
      const boards = (await api(await get(base + '/figure-boards'))).boards;
      await select(reuse, 'Saved figure for template / output', new RegExp(boards[0].id.slice(0, 8)));
      await reuse.getByRole('button', { name: 'Preview included output files', exact: true }).click();
      await expect(reuse.getByTestId('output-preview')).toContainText('panels/A-source.csv');
      await reuse.getByRole('textbox', { name: 'Figure legend (Chinese)', exact: true }).fill('桌面验证图注');
      await reuse.getByRole('checkbox', { name: 'I reviewed the included data, identifiers and notes' }).check();
      await reuse.getByRole('button', { name: 'Save frozen output', exact: true }).click();
      const zipLink = reuse.getByRole('link', { name: /Download research ZIP/ });
      await expect(zipLink).toBeVisible();
      const href = await zipLink.getAttribute('href');
      const bytes = await artifact(href, 'zip', 'D1-frozen-research');
      await page.reload(); await expect(zipLink).toBeVisible();
      expect(hash(await (await get(href)).body())).toBe(hash(bytes));
      check('template-replacement-and-frozen-zip-reopen');
      await artifact(`/api/v1/projects/${projectId}/exports/cleaned-data.csv`, 'csv', 'D1-cleaned-data');
      await page.getByRole('button', { name: 'Switch to overview A interface', exact: true }).click();
      await expect(page.getByRole('button', { name: 'Switch to classic interface', exact: true })).toBeVisible();
      await page.getByRole('button', { name: 'Switch to classic interface', exact: true }).click();
      await page.getByRole('button', { name: 'Switch to dark theme', exact: true }).click();
      await page.getByRole('button', { name: 'Switch to light theme', exact: true }).click();
      await page.getByLabel('Language', { exact: true }).click();
      await page.getByRole('option', { name: '中文', exact: true }).click();
      await expect(page.locator('html')).toHaveAttribute('lang', 'zh');
      await post('/api/locale', { data: { locale: 'en' } }); await page.reload();
      check('interface-modes-themes-and-chinese');
      await page.goto(origin + '/history');
      expect((await api(await get('/api/v1/projects'))).projects.length).toBeGreaterThanOrEqual(3);
      check('local-history');
      await page.goto(origin + '/');
      await page.getByRole('button', { name: 'View examples', exact: true }).click();
      await page.locator('[data-example-slug="surface-3d"]').getByRole('button', { name: 'Open this example', exact: true }).click();
      await expect(page.getByRole('heading', { name: 'Confirm your data', exact: true })).toBeVisible();
      await page.getByRole('button', { name: 'Review data quality', exact: true }).click();
      const keep = page.getByRole('button', { name: 'Keep all remaining as is', exact: true });
      if (await keep.isVisible()) await keep.click();
      await page.getByRole('button', { name: 'Create chart', exact: true }).click();
      await expect(page.getByRole('combobox', { name: 'Chart type', exact: true })).toHaveText('3D surface');
      await expect.poll(() => page.evaluate(() => (window.__labvizD1WebGL ?? []).some(entry => entry.canvas.isConnected && entry.canvas.width > 0 && entry.canvas.height > 0 && !entry.context.isContextLost())), { timeout: 60000 }).toBe(true);
      report.webgl = await page.evaluate(() => window.__labvizD1WebGL.filter(entry => entry.canvas.isConnected).map(entry => ({ kind: entry.kind, width: entry.canvas.width, height: entry.canvas.height, contextLost: entry.context.isContextLost() })));
      await page.getByRole('button', { name: 'Rotate right', exact: true }).click();
      await page.getByRole('button', { name: 'Zoom in', exact: true }).click();
      await page.screenshot({ path: path.join(trial, 'webgl-surface.png') });
      check(config.shell==='headless-test-adapter'?'headless-browser-webgl-surface':'actual-webview2-webgl-surface');
      await page.goto(origin + `/workspace/${projectId}`);
    }
  }
  phase('final-idle'); await pause(3000);
  expect(report.pageErrors).toEqual([]);
  expect(report.externalRequests).toEqual([]);
  report.passed = true; save();
  fs.writeFileSync(path.join(trial, 'export-links.json'), JSON.stringify(report.exports, null, 2));
  console.log(JSON.stringify({ passed: true, checks: report.checks.length, exports: report.exports.length }));
  process.exit(0);
} catch (error) {
  report.error = String(error.stack ?? error); save();
  if (page) { try { await page.screenshot({ path: path.join(trial, `failure-${mode}.png`) }); } catch {} }
  console.error(report.error); process.exit(1);
}
