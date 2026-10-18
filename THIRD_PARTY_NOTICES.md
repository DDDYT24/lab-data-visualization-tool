# Third-party components in the Windows RC

LabViz source uses the MIT license. Bundled dependencies retain their own licenses.

- Node.js 24.22.0: `runtime/node/LICENSE`, including upstream dependency notices.
- CPython 3.13.16: `runtime/python/LICENSE.txt`.
- Python packages: their `runtime/python/Lib/site-packages/*.dist-info` license/notice
  files, retained from the audited wheel installation.
- Production website dependencies: `licenses/WEB-THIRD-PARTY-LICENSES.txt`, collected
  from the frozen lock and dependency license files. The notice collection may also
  list optional/platform packages not traced into this Windows runtime.
- Noto Sans SC font: `V2.0/assets/fonts/OFL.txt`.

`V2.0/packaging/windows-v23/collect-web-licenses.mjs` regenerates the web notice collection.
Runtime downloads and hashes are pinned in `V2.0/packaging/windows-v23/runtimes.json`.
The Inno Setup compiler is a build tool and is not bundled in the installed application.

The V2.2 release retains its original component versions and notices in its tagged source and installer.
