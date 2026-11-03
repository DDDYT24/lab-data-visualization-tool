import React from 'react';
import { renderToString } from 'react-dom/server';
import { NextIntlClientProvider } from 'next-intl';
import { describe, expect, it } from 'vitest';
import { DesktopSession } from './desktop-session';

function render(localRuntime: boolean) {
  return renderToString(React.createElement('section',null,<NextIntlClientProvider locale="en" timeZone="UTC" messages={{}}><DesktopSession localRuntime={localRuntime}><p>Original browser content</p></DesktopSession></NextIntlClientProvider>));
}

describe('desktop recovery does not replace ordinary browser server rendering',()=>{
  it('keeps the regular website available in the server response',()=>{
    expect(render(false)).toContain('Original browser content');
  });
  it('waits before mounting local import hooks that need reconstructed source files',()=>{
    const html=render(true);
    expect(html).not.toContain('Original browser content');
    expect(html).toContain('Restoring local workspace');
  });
});
