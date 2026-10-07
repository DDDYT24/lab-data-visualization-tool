# Third-party components in the Windows installer

LabViz source uses the MIT license. Bundled dependencies retain their own licenses;
the installer is not relicensed as a single MIT-only work.

- Node.js 24.17.0: `licenses/NODE-LICENSE.txt`, including upstream dependency notices.
  Source: https://github.com/nodejs/node/blob/v24.17.0/LICENSE
- CPython 3.13.7: `runtime/python/LICENSE.txt`.
- Python packages: their `runtime/python/Lib/site-packages/*.dist-info` license/notice
  files, retained from the audited wheel installation.
- Production website dependencies: `licenses/WEB-THIRD-PARTY-LICENSES.txt`, collected
  from the frozen lock and dependency license files. The notice collection may also
  list optional/platform packages not traced into this Windows runtime.
- Noto Sans SC font: `V2.0/assets/fonts/OFL.txt`.

`V2.0/packaging/windows/collect-web-licenses.mjs` regenerates the web notice collection.
The Inno Setup compiler is a build tool and is not bundled in the installed application.
