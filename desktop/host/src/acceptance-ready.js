(() => {
  const config = __LABVIZ_ACCEPTANCE_READY__;
  if (window.top !== window || location.origin !== config.origin) return;
  let sent = false;
  const visible = element => !!element && element.isConnected && element.getClientRects().length > 0
    && getComputedStyle(element).visibility !== 'hidden' && !element.matches(':disabled');
  function editor() {
    if (location.pathname !== config.pathname || document.visibilityState !== 'visible'
      || document.readyState === 'loading' || typeof window.labvizDesktopCheckpoint !== 'function'
      || window.labvizDesktop?.profileId !== config.profileId) return null;
    if (config.kind === 'home') {
      const root = document.querySelector('[data-testid="home-dropzone"]');
      const button = root?.querySelector('button');
      return visible(button) && root.querySelector('input[type="file"]') ? button : null;
    }
    return [...document.querySelectorAll('input')].find(element => visible(element) && !element.readOnly
      && element.value === config.expectedTitle
      && (element.getAttribute('aria-label') === 'Figure title'
        || [...(element.labels ?? [])].some(label => label.textContent?.trim() === 'Figure title'))) ?? null;
  }
  function poll() {
    if (sent) return;
    if (!editor()) { requestAnimationFrame(poll); return; }
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (!editor()) { requestAnimationFrame(poll); return; }
      sent = true;
      const signal = {
        id: config.id, sequence: config.sequence, kind: config.kind,
        rendererUtcSeconds: (performance.timeOrigin + performance.now()) / 1000,
        documentElapsedMs: performance.now(), focused: document.hasFocus(),
      };
      // This acceptance-only command records timing. It cannot navigate or
      // mutate the editor, and rejects ordinary (non-acceptance) launches.
      window.__TAURI_INTERNALS__.invoke('acceptance_editor_ready', {signal}).catch(error => {
        window.__labvizAcceptanceReadinessError = String(error);
      });
    }));
  }
  requestAnimationFrame(poll);
})();
