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
expect(page).toBeTruthy();
page.setDefaultTimeout(30000);
const metadata = await page.evaluate(() => window.labvizDesktop);
expect(metadata.profileId).toBe(services.profileId);
expect(metadata.apiBase).toBe(services.webUrl + '/api/v1');
expect(metadata.automaticSuspension).toBe(false);
if (action === 'draft') {
  await page.getByRole('button', { name: 'Research workspace', exact: true }).click();
  await page.getByRole('tab', { name: 'Figures', exact: true }).click();
  await page.getByRole('textbox', { name: 'Figure title', exact: true }).fill('D2 unsaved single-instance draft');
} else if (action === 'wake') {
  await expect(page.getByRole('textbox', { name: 'Figure title', exact: true })).toHaveValue('D2 unsaved single-instance draft');
} else if (action === 'preferences') {
  await page.goto(services.webUrl + '/');
  await page.getByRole('button', { name: 'Switch to dark theme', exact: true }).click();
  await page.getByRole('button', { name: 'Switch to overview A interface', exact: true }).click();
  await page.request.post(services.webUrl + '/api/locale', {data: {locale:'zh'}});
  await page.reload();
  await expect.poll(() => JSON.parse(fs.readFileSync(path.join(trial, 'profile/preferences.json'))).locale).toBe('zh');
  await expect.poll(() => JSON.parse(JSON.parse(fs.readFileSync(path.join(trial, 'profile/preferences.json'))).storage['labviz:user-preferences:v1']).appearance).toBe('dark');
  expect(JSON.parse(fs.readFileSync(path.join(trial, 'profile/preferences.json'))).storage['labviz:v23:interface-mode:v1']).toBe('overview');
} else if (action === 'restored') {
  await expect.poll(() => page.evaluate(() => document.documentElement.lang)).toBe('zh');
  const stored = await page.evaluate(() => ({theme:JSON.parse(localStorage.getItem('labviz:user-preferences:v1')).appearance,mode:localStorage.getItem('labviz:v23:interface-mode:v1')}));
  expect(stored).toEqual({theme:'dark',mode:'overview'});
  const old = JSON.parse(fs.readFileSync(path.join(trial, 'probe-preferences.json')));
  expect(old.origin).not.toBe(services.webUrl);
  expect(old.metadata.profileId).toBe(metadata.profileId);
} else if (action === 'internal-window') {
  const count = browser.contexts()[0].pages().length;
  await page.evaluate(() => window.open('/help','_blank'));
  await expect.poll(() => new URL(page.url()).pathname).toBe('/help');
  expect(browser.contexts()[0].pages()).toHaveLength(count);
}
fs.writeFileSync(path.join(trial, `probe-${action}.json`),JSON.stringify({passed:true,action,origin:services.webUrl,metadata},null,2));
console.log(JSON.stringify({passed:true,action}));
process.exit(0);
