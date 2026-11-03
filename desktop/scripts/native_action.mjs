import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const root = path.resolve(import.meta.dirname, '../..');
const require = createRequire(path.join(root, 'V2.0/web/package.json'));
const { chromium, expect } = require('@playwright/test');
const trial = path.resolve(process.argv[2]);
const action = process.argv[3];
const config = JSON.parse(fs.readFileSync(path.join(trial, 'launcher.json')));
const services = JSON.parse(fs.readFileSync(path.join(trial, 'services.json')));
const browser = await chromium.connectOverCDP(`http://127.0.0.1:${config.debugPort}`);
const page = browser.contexts()[0].pages().find(p => p.url().startsWith(services.webUrl));
if (!page) throw new Error('Owned WebView page is missing');
const session = await browser.newBrowserCDPSession();
await session.send('Browser.setDownloadBehavior', { behavior: 'default', eventsEnabled: false });
await session.detach();
if (action === 'home') {
  await page.goto(services.webUrl + '/');
  await page.locator('input[type=file]').first().waitFor({ state: 'attached' });
} else if (action === 'exports') {
  const entries = JSON.parse(fs.readFileSync(path.join(trial, 'export-links.json')));
  const id = entries[0].href.match(/\/projects\/([^/]+)/)[1];
  await page.goto(services.webUrl + '/workspace/' + id);
  const research = page.getByRole('button', { name: 'Research workspace', exact: true });
  await expect(research).toBeVisible();
  await research.click();
  await page.getByRole('tab', { name: 'Analysis', exact: true }).click();
  const charts = page.getByTestId('research-charts');
  await charts.getByRole('combobox', { name: 'Saved charts', exact: true }).click();
  await page.getByRole('option', { name: /D1 Raw/ }).last().click();
  await expect(charts.getByRole('link', { name: 'Download PNG', exact: true })).toBeVisible();
} else if (action === 'download') {
  const format = process.argv[4];
  const exports = JSON.parse(fs.readFileSync(path.join(trial, 'export-links.json')));
  const entry = exports.find(item => item.format === format);
  if (!entry) throw new Error('No verified artifact for ' + format);
  fs.mkdirSync(path.join(trial, 'native'), { recursive: true });
  fs.writeFileSync(path.join(trial, 'native-pending.json'), JSON.stringify({ format, expectedSha256: entry.sha256, expectedBytes: entry.bytes, target: path.join(trial, 'native', 'actual.' + format) }, null, 2));
  console.log('Native save dialog requested for ' + format);
  try {
    await page.goto(new URL(entry.href, services.webUrl).toString(), { timeout: 120000 });
  } catch (error) {
    if (!/ERR_ABORTED|Download is starting|Download started/.test(String(error))) throw error;
  }
} else if (action === 'surface') {
  await page.goto(services.webUrl + '/');
  await page.getByRole('button', { name: 'View examples', exact: true }).click();
  await page.locator('[data-example-slug="surface-3d"]').getByRole('button', { name: 'Open this example', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Confirm your data', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Review data quality', exact: true }).click();
  const keep = page.getByRole('button', { name: 'Keep all remaining as is', exact: true });
  if (await keep.isVisible()) await keep.click();
  await page.getByRole('button', { name: 'Create chart', exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'Chart type', exact: true })).toHaveText('3D surface');
  await page.locator('canvas').first().waitFor({ state: 'visible' });
} else if (action === 'verify-import') {
  await expect(page.getByRole('heading', { name: 'Confirm your data', exact: true })).toBeVisible();
  const projectId = new URL(page.url()).pathname.split('/').at(-1);
  const response = await page.request.get(services.webUrl + `/api/v1/projects/${projectId}/workspace`);
  if (!response.ok()) throw new Error('Native import did not produce a readable project');
  const workspace = await response.json();
  expect(workspace.preview.totalRows).toBe(12);
  fs.writeFileSync(path.join(trial, 'native-import-verification.json'), JSON.stringify({ passed: true, source: workspace.session.source.filename, rows: workspace.preview.totalRows, page: new URL(page.url()).pathname, nativePickerObserved: true }, null, 2));
} else {
  throw new Error('Unknown native action');
}
console.log('Native action ready');
process.exit(0);
