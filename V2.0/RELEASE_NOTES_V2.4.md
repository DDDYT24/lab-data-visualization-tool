# LabViz V2.4.0-rc.1

A Windows desktop preview of V2.4. V2.3.0 remains the stable release while this candidate is evaluated.

V2.4 gives every chart one drawing plan. The server works out every number that is drawn, and the preview in the window and the exported file draw that same plan, so the two no longer disagree.

## What is new since V2.3.0

**Preview equals export.**
- Line, scatter, bar, histogram, box, heatmap and 3D surface charts preview and export from one plan, with the same axis ranges, ticks and grid.
- So do research charts and the panels of composed figures.
- The browser side no longer bins, takes quantiles or builds charts from raw rows.

**Interactive research charts.**
- Hover a newly saved research chart to read values, zoom it, and click a point to find its source row.
- Charts with more than 20,000 observations, and charts saved before V2.4, still show their saved image.

**Exports draw every row.**
- Before, line, scatter and bar exports silently drew at most 250 points. They now draw all rows; the preview still thins to 250 points and keeps the peaks.
- 3D surface exports draw the whole grid; they used to sample at most 2,000 points and could miss a spike.
- The heatmap colour range is fixed at −1 to 1 with the first row on top.
- Turning the grid off now also applies to surface exports.

**Publication figures.**
- Figures hold up to six panels, in a grid or a mosaic. Three-panel figures are allowed.
- Each panel can have log10 or symlog axes, a manual range and manual ticks.
- Panels take annotations: text, reference lines, shaded ranges and brackets.
- One panel can overlay several saved charts of the same data, for example raw points over a violin.
- Figures from V2.3 open unchanged. Saving a V2.3 figure again produces the same PNG, SVG and PDF files.

**Editing.**
- Chart settings are split into Data, Method and Appearance tabs.
- Appearance changes such as colours, labels and fit-label switches redraw without recalculating.
- Figure layouts can be undone and redone.
- Explanatory text that repeated what the screen already shows has been removed.

**Safety.**
- A figure too large to draw (over 32,768 px a side or 200 megapixels) is refused before anything is saved.
- Axis limits at the edge of the number range are refused instead of failing.
- A figure that could not be built because a font was missing or the service was busy can be tried again.
- If an interactive view cannot load, the window says so.

## Data and the way back

V2.4 adds two tables to the local database. A profile that V2.4.0-rc.1 has opened is marked with data epoch 3. V2.3.0 refuses that profile without changing it.

To keep a way back:
1. Before installing, make a backup from the data tools in V2.3.0.
2. To return, reinstall V2.3.0 and restore that backup.

See [installation, backup and recovery](docs/INSTALL_V2.3.md).

## Install

Download the setup file and `SHA256SUMS.txt` from the release, check the SHA-256, and run the setup file. The installer is **not code-signed**, so Windows SmartScreen warns before it runs; keep Windows security controls enabled. It installs over V2.3.0 in place, bundles Python, Node.js, the scientific libraries and fonts, and uses the Microsoft Edge WebView2 runtime. Uninstalling keeps your data.

## Known issues

- The installer is unsigned.
- Line, scatter and bar exports with more than 250 rows differ from V2.3.0, because every row is now drawn. A 100,000-row bar export takes about 3 to 6 seconds.
- In a narrow custom figure canvas, category names and the count line under them can overlap. The default size is unaffected.
- A paired chart without a group column lists one legend entry named "Undeclared".
- Messages that refuse a chart or a figure are in English only.
- Selecting points on a chart needs a mouse; the data table rows are the keyboard path.
- Violin densities can differ between computers in their last digits, so a violin's plan identifier can differ while the drawing looks the same.
- picomatch has a published advisory with no fixed release. It reaches only an unused part of one production package and development tools that are not shipped; see the [dependency review](docs/DEPENDENCY_SECURITY_REVIEW.md).
- Carried over from V2.3.0 and still not checked:
  - a keyboard-only pass of every screen;
  - a clean machine that has never had LabViz;
  - a physically disconnected workflow;
  - screen-reader use.
- Native macOS and Linux installers and hosted collaboration are outside this release.
