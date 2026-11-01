(() => {
  const configuration = __LABVIZ_DESKTOP_CONFIGURATION__;
  if (window.location.origin !== configuration.origin) return;
  const marker = 'labviz:desktop-profile:v1';
  const keys = ['labviz:user-preferences:v1', 'labviz:v23:interface-mode:v1'];
  const preferences = configuration.preferences;
  const locale = () => document.cookie.match(/(?:^|;\s*)labviz-locale=(en|zh)(?:;|$)/)?.[1] ?? 'en';
  let restoredLocale = false;
  try {
    const storageGeneration = configuration.profileId + ':' + (preferences.generation ?? 'legacy');
    if (localStorage.getItem(marker) !== storageGeneration) {
      for (const key of keys) {
        if (Object.hasOwn(preferences.initial.storage, key)) localStorage.setItem(key, preferences.initial.storage[key]);
      }
      localStorage.setItem(marker, storageGeneration);
      if (!document.cookie.includes('labviz-locale=')) {
        document.cookie = `labviz-locale=${preferences.initial.locale}; path=/; SameSite=Lax`;
        restoredLocale = true;
      }
    }
  } catch { /* The interface remains usable when browser storage is unavailable. */ }
  Object.defineProperty(window, 'labvizDesktop', { value: Object.freeze({ version: 1, appId: 'org.labviz.desktop.preview', profileId: configuration.profileId, apiBase: configuration.origin + '/api/v1', closeBehavior: 'tray', automaticSuspension: true }), configurable: false });
  Object.defineProperty(window, 'labvizDesktopTransport', { value: Object.freeze(preferences), configurable: false });
  const originalFetch = window.fetch.bind(window);
  const current = () => ({ format: 'labviz-desktop-preferences-v1', profileId: configuration.profileId, locale: locale(), storage: Object.fromEntries(keys.map(key => [key, localStorage.getItem(key)]).filter(([, value]) => value !== null)) });
  let previous = JSON.stringify(preferences.initial);
  let saving = false;
  window.labvizDesktopFlushPreferences = async () => {
    while (saving) await new Promise(resolve => setTimeout(resolve, 25));
    saving = true;
    try {
      const encoded = JSON.stringify(current());
      const response = await originalFetch(preferences.url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-LabViz-Desktop': preferences.token }, body: encoded, signal: AbortSignal.timeout(30000) });
      if (!response.ok) throw new Error('Preferences could not be saved. The workspace will remain open.');
      previous = encoded;
    } finally { saving = false; }
  };
  async function persist() {
    if (saving || window.location.hash.includes('labviz-access=')) return;
    try {
      const value = current();
      const encoded = JSON.stringify(value);
      if (encoded === previous) return;
      saving = true;
      const response = await originalFetch(preferences.url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-LabViz-Desktop': preferences.token }, body: encoded, keepalive: true });
      if (response.ok) previous = encoded;
    } catch { /* A later preference event retries the same bounded snapshot. */ }
    finally { saving = false; }
  }
  window.addEventListener('labviz:preferences-changed', persist);
  window.addEventListener('labviz:interface-mode-changed', persist);
  window.addEventListener('pagehide', persist);
  let pendingRequests = 0;
  window.labvizDesktopPending = () => pendingRequests;
  window.fetch = async (...arguments_) => {
    const target = new URL(typeof arguments_[0] === 'string' ? arguments_[0] : arguments_[0].url ?? String(arguments_[0]), location.href);
    const tracked = target.origin === configuration.origin && target.pathname.startsWith('/api/');
    if (tracked) pendingRequests++;
    try {
      const response = await originalFetch(...arguments_);
      if (target.pathname === '/api/locale' && response.ok) queueMicrotask(persist);
      return response;
    } finally { if (tracked) pendingRequests--; }
  };
  setInterval(persist, 1000);
  window.addEventListener('DOMContentLoaded', () => {
    if (restoredLocale && document.documentElement.lang !== preferences.initial.locale) window.location.reload();
  }, { once: true });
})();
