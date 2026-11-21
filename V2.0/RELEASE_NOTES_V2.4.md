# LabViz V2.4.0

The Windows desktop release of V2.4. It follows V2.3.0.

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
- Exported PDFs embed TrueType fonts and SVGs keep their text as text, so journals that refuse Type 3 fonts accept them and the text stays editable. Chinese labels carry their glyphs in the SVG. PNG output is unchanged.
- A figure's text can be set in DejaVu Sans (the default, as before), Arial or Times New Roman.
- Figures no longer print the input rows, revision and method lines under each panel; they stay in the output notes, and a switch prints them again. The count line under category names can be turned off.
- Figures from V2.3 open unchanged. Saved again, they get the embedded fonts and lose the footer; the drawing is otherwise the same.
- Fixed: in a classic chart of more than 250 rows captured into a figure, most points linked to the wrong source row. Captures saved earlier keep their stored links.

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

V2.4 adds two tables to the local database. A profile that V2.4.0 has opened is marked with data epoch 3. V2.3.0 refuses that profile without changing it.

Installing over V2.3.0 keeps the V2.3.0 program and a snapshot of your data taken just before the upgrade. To go back, exit LabViz and run this in PowerShell:

```powershell
$root = "$env:LOCALAPPDATA\Programs\LabVizDesktopPreview"
$id = (Get-Content "$root\current-candidate.json" | ConvertFrom-Json).packageId
$v = "$root\versions\$($id.Substring(0, 24))"
& "$v\app\runtime\python\python.exe" -B -X utf8 "$v\runtime\installation.py" rollback --install-root $root --profile-base "$env:LOCALAPPDATA\LabVizDesktopPreview" --payload "$v\app"
```

This makes V2.3.0 the active program again and restores the snapshot. When it worked, it prints one line that contains `"activated": true` and `"explicitRollback": true`. Any other result, an error or exit code 20, means V2.3.0 did not become active; keep the output when you report it. Work saved in V2.4 is not lost: it is kept beside the restored data in a folder whose name starts with `data.retained-`, together with a backup of it.

## Install

Download the setup file and `SHA256SUMS.txt` from the release, check the SHA-256, and run the setup file. The installer is **not code-signed**, so Windows SmartScreen warns before it runs; keep Windows security controls enabled. It installs over V2.3.0 in place, bundles Python, Node.js, the scientific libraries and fonts, and uses the Microsoft Edge WebView2 runtime. Uninstalling keeps your data.

## Known issues

- The installer is unsigned.
- Line, scatter and bar exports with more than 250 rows differ from V2.3.0, because every row is now drawn. A 100,000-row bar export takes about 3 to 6 seconds.
- With 100,000 rows per panel, composed figures are slower than their targets: a four-panel export takes about 17 seconds and a six-panel one about 25 (targets 8 and 10 seconds), and recalculating a strip or violin chart takes 5 to 6 seconds (target 3). At 10,000 rows every target is met.
- Reopening a saved project of 100,000 rows with a four-panel figure took about 3.5 seconds in testing, and once more than 5; V2.3.0 took about 1.3 seconds. The cause is not known yet.
- With the footer switched on, a narrow custom figure canvas can overlap category names, the count line and the footer.
- With the footer off, a figure keeps the space the footer used at its bottom; at the default canvas size about a fifth of a one-row figure is empty.
- Chinese labels are drawn in a thin weight of the bundled font.
- In the figure editor the interactive panel views draw text in the app font; the composed figure shows the chosen font.
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
