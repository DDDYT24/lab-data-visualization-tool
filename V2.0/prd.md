# LabViz V2.0 Product Requirements Document

**Status:** V2.0 website requirements implemented; V2.1 structured-3D and research-workflow
extension shipped in the V2.1.1 local release. Future work is tracked only in [`TODO.md`](TODO.md).
**Primary prototype tool:** Figma
**Primary language of the interface:** English
**Secondary interface language:** Simplified Chinese
**Target platform for V2.0:** Desktop-first responsive web application; desktop packaging is deferred
**Document audience:** Product design, UI/UX design, frontend engineering, backend engineering, and Figma collaborators
**Prototype brand asset:** `assets/labviz-logo.png`
**Synthetic table reference:** `assets/labviz-synthetic-table.png`

## 1. Figma Design Brief

Design a high-fidelity, light-mode product prototype for **LabViz V2.0**, a local-first experimental data visualization tool for people who do not know Python and have little or no data-analysis experience.

Use the provided `assets/labviz-logo.png` horizontal logo in the application shell. Preserve the exact wordmark `LabViz`; do not redraw it, add a slogan, or introduce a competing brand symbol.

The main user should be able to upload an Excel file containing thousands of experimental measurements, review data-quality issues, safely clean the data, create publication-ready charts, customize titles, units, colors, error bars, and fitted curves, and export the result without writing code.

The experience should feel:

- Professional and scientifically credible.
- Friendly and understandable to a beginner.
- Focused on producing publication-ready figures.
- Calm, precise, and lightweight rather than enterprise-heavy.

Do not make the interface look like a generic business-intelligence dashboard, a spreadsheet clone, a futuristic software product, or a dark cyber interface. Avoid decorative gradients, glassmorphism, oversized marketing sections, and excessive card nesting.

## 2. Product Definition

### Product statement

LabViz helps non-programmers turn experimental spreadsheet data into clear, publication-ready visualizations through a guided workflow:

```text
Upload data → Inspect quality → Clean safely → Create charts → Export
```

### Primary users

1. Laboratory staff with no Python experience and beginner-level data-analysis skills.
2. University laboratory students.
3. Academic researchers.
4. Research and development engineers.

### Primary user scenario

A user uploads an Excel file containing thousands of experimental records. LabViz identifies missing values and suspicious data points, explains why each point may be problematic, lets the user approve cleaning actions, creates multiple curves, allows figure customization, and exports a publication-ready image.

### Product principles

- No Python, terminal, or programming knowledge is required.
- The fastest path should produce a useful chart with sensible defaults.
- Advanced controls appear through progressive disclosure.
- Suspicious data is highlighted and explained, never changed automatically.
- Every cleaning action is reversible.
- The original source file remains unchanged.
- Local processing and privacy are clearly communicated.
- Statistical choices must not imply that a paid option is more scientifically correct.
- Desktop supports the complete workflow; mobile focuses on viewing, sharing, and simple edits.

### V2.1 product direction

V2.1 keeps the V2.0 local-first workflow and adds a safer path for structured 3D experimental data.
The product may recommend a chart based on the shape of the uploaded data, but it must not silently
change the chart type, cleaning decisions, or source data. A regular X/Y grid should be explained
as a candidate surface, while incomplete, duplicated, irregular, or collinear points must receive a
plain-language diagnostic and a suitable surface, heatmap, or scatter fallback.

V2.1 also adds an optional repeated-acquisition identity layer. Before upload, a user may provide
an experiment name plus acquisition, replicate, and batch labels. Each file becomes a distinct
physical `ExperimentRun`; saved history and publication exports retain that lineage. A software
`ProcessingRun` continues to describe parse, profile, clean, analyze, or export execution only.
Experiment dashboards, cross-experiment statistics, and team roles remain outside this slice.

The V2.1 execution order, acceptance gates, and future backlog are maintained only in
[`TODO.md`](TODO.md#v21-roadmap). [`PROJECT_PLAN.md`](PROJECT_PLAN.md) is the architecture
reference and does not own completion status.

## 3. Business and Deployment Model

### Business model

LabViz uses an **open-core plus paid cloud services** model.

- Core local functionality remains openly available.
- Future revenue may come from cloud storage, synchronization, sharing controls, larger cloud workloads, workflow automation, advanced templates, and team features.
- Exact pricing and paid entitlements are not finalized.
- The V2.0 prototype must not include a pricing page or checkout flow.
- Core statistical correctness, including 90%, 95%, and 99% confidence intervals, must not be presented as a paid accuracy upgrade.

### Future Windows local mode

The V2.0 implementation is website-first. The following local-mode requirements remain approved for a future desktop release but do not block the initial website:

- Windows is the only desktop installer required for the first V2.0 release.
- No login is required.
- Projects and history are stored on the user's computer.
- A compressed project copy allows a project to reopen after the original file is moved or deleted.
- Users can inspect storage use and clear individual or all local history.
- Local single-file limit: 200 MB.

### Cloud mode

- Guests can use the product temporarily without an account.
- Guest projects are not added to history.
- Guest temporary data expires two hours after the last activity.
- Uploaded original files should be deleted as soon as parsing is complete.
- Temporary processed data and results are deleted when the session expires.
- Email one-time-code authentication is the only required login method for V2.0.
- Signed-in users may explicitly save projects to the cloud for history, cross-device access, and sharing.
- Default cloud single-file limit: 50 MB.

### Sharing defaults

- A shared link opens a read-only interactive chart.
- It includes the chart title and optional experiment description.
- It does not expose the original uploaded file.
- Downloading is disabled by default.
- The creator may explicitly allow PNG, SVG, or PDF downloads.
- The creator can disable the link at any time.

## 4. Supported Data and File Strategy

### File formats

- Excel: `.xlsx`
- CSV: `.csv`
- TSV: `.tsv`
- Delimited text: `.txt`
- JSON: `.json`

### Local file-size behavior

| File size | Required behavior |
| --- | --- |
| 0–50 MB | Process normally and completely. |
| More than 50 MB and up to 200 MB | Automatically enable Large File Mode. |
| More than 200 MB | Do not open the file. Explain the limit and provide splitting guidance. |

### Large File Mode

- Show a visible but calm “Large File Mode” indicator.
- Use a virtualized or bounded table preview instead of rendering all rows.
- Use deterministic, evenly distributed sampling for chart previews.
- Apply data cleaning to the complete dataset.
- Export the complete cleaned dataset.
- Process long operations in the background.
- Show real progress and the current processing stage.

### V2.1 structured surface data

- Treat three numeric columns as a surface candidate only after checking the selected X/Y pair.
- Report the number of unique X values, unique Y values, usable points, duplicate `(X,Y)` pairs,
  missing grid cells, non-finite values, and whether the points are collinear.
- For a complete rectangular grid, show the detected dimensions and offer a user-confirmed 3D
  surface recommendation.
- In the 3D editor, expose explicit X, Y, and Z selectors. The first surface coordinate is Y and
  the response height is Z; generic multi-series labels are not sufficient for this mode.
- For incomplete or irregular points, explain why a regular surface may be misleading and offer a
  heatmap or scatter fallback where the data supports it.
- Keep the original row order and source data unchanged. Grid interpretation belongs to chart
  analysis, not destructive cleaning.
- Never leave the page looking frozen.

## 5. Information Architecture

### Primary routes

| Route | Screen | Purpose |
| --- | --- | --- |
| `/` | Home | Start a new analysis immediately and access recent projects. |
| `/workspace/new` | New workspace | Upload and initialize a new dataset. |
| `/workspace/[id]` | Data workspace | Inspect, clean, visualize, and export. |
| `/history` | History | Reopen, duplicate, export, share, or delete saved projects. |
| `/share/[token]` | Shared chart | View a responsive, read-only interactive chart. |
| `/settings` | Settings | Manage language, export defaults, privacy, and local storage. |
| `/help` | Help | Access onboarding, sample data, and troubleshooting. |

Authentication should use a modal or lightweight focused screen rather than a permanent primary-navigation destination.

## 6. Global Application Shell

### Desktop shell

- Top bar: LabViz mark, current project name, local/cloud indicator, save state, language switcher, account menu.
- Left side: primary navigation on Home and History; four-step workflow navigation inside the workspace.
- Main area: data table or chart canvas.
- Right inspector: contextual controls for the active step.
- Bottom status bar in the workspace: total rows, valid rows, suspicious points, sampling state, and processing state.

### Global behavior

- Use clear labels before icons for beginner-facing actions.
- Keep one obvious primary action per screen.
- Use tooltips to explain statistical terms in plain language.
- Preserve undo and redo throughout cleaning and chart editing.
- Warn before losing unsaved work.
- Distinguish Local, Temporary Cloud, and Saved Cloud states visibly.
- Use autosave only where storage is available; never imply a guest session is permanently saved.

## 7. Screen Requirements

### 7.1 Home

#### Purpose

Let a first-time user begin by uploading a file without reading documentation.

#### Modules

1. Compact top navigation with language and login controls.
2. Hero upload area with drag-and-drop and file picker.
3. Short three-step explanation: Upload, Review, Export.
4. Sample dataset action.
5. Supported formats and privacy note.
6. Recent local or cloud projects when available.
7. A Windows desktop download entry only after the deferred desktop application exists.

#### Empty state

- Large upload target.
- Friendly line: “Drop your experiment file here.”
- Secondary action: “Try sample data.”
- Show supported formats and the relevant file-size limit.

#### Returning-user state

- Keep upload as the primary action.
- Add a compact recent-project section below it.
- Do not turn the page into a dense project dashboard.

### 7.2 Workspace Step 1 — Import Data

#### Modules

- File summary: name, type, size, local/cloud processing status.
- Excel sheet selector.
- Header-row detection and override.
- Column preview.
- Detected column types.
- Unit detection or manual unit entry.
- Confirmed data-range preview.

#### Required states

- Idle upload.
- Drag active.
- Validating file.
- Uploading to temporary cloud processing.
- Reading workbook.
- Selecting an Excel sheet.
- Detecting headers and column types.
- Unsupported format.
- Password-protected or unreadable workbook.
- Empty workbook or empty sheet.
- File above the permitted limit.
- Large File Mode activated.

### 7.3 Workspace Step 2 — Inspect and Clean

#### Modules

- Quality summary with total rows, missing values, duplicates, type conflicts, and suspicious points.
- Filterable issue list.
- Data table with highlighted cells and rows.
- Plain-language explanation panel for the selected issue.
- Suggested action with explicit user confirmation.
- Before-and-after impact preview.
- Undo, redo, and restore-original actions.

#### Issue types

- Missing values.
- Duplicate rows.
- Text inside numeric columns.
- Extreme values.
- Sudden changes in a sequence.
- Values outside a user-defined valid range.
- Values inconsistent with the overall trend.

#### Interaction rules

- Never automatically delete or replace a suspicious value.
- Do not preselect destructive actions.
- Explain the detection reason and affected row count.
- Allow individual and bulk decisions.
- Separate “Ignore,” “Exclude from chart,” and “Remove from cleaned copy.”
- Keep the original source untouched.

#### Required states

- Profiling in progress.
- No issues found.
- Issues found.
- One issue selected.
- Bulk selection active.
- Cleaning in progress.
- Cleaning complete.
- Cleaning failed with a recoverable explanation.

### 7.4 Workspace Step 3 — Create Chart

#### Chart types

- Line chart.
- Scatter plot.
- Bar chart.
- Histogram.
- Box plot.
- Correlation heatmap.
- 3D surface chart.
- Error bars.
- Fitted curves.
- Dual Y axes.
- Multi-panel figure with up to four panels in the core V2.0 interface.

Future cloud plans may allow six panels, but the initial prototype should focus on a clear four-panel workflow and must not design a payment interruption.

#### Fitted curves

- Linear.
- Polynomial, with a maximum order of three by default.
- Exponential.
- Logarithmic.
- Power function.
- Optional equation display.
- Optional R² display.
- Optional confidence band.
- Explain when a model is not suitable or cannot be fitted.

#### Error bars and uncertainty

- Standard error.
- Confidence interval levels: 90%, 95%, and 99%.
- Default confidence interval: 95%.
- Allow a user to select an existing error column when present.
- Explain that a higher confidence level normally creates a wider interval and is not automatically “more accurate.”

#### Modules

- Chart-type recommendations based on selected columns.
- X, Y, grouping, error, and panel field selectors.
- For `3D surface`, explicit X, Y, and Z selectors with the detected grid summary beside them.
- A non-destructive explanation when the current row order would make a line chart join repeated
  X values or different surface slices.
- Large central chart canvas.
- Right-side appearance and analysis controls.
- Data-point inspection on hover or selection.
- Add-panel and arrange-panel controls.

#### Required states

- No numeric columns available.
- No fields selected.
- Recommended chart ready.
- Rendering.
- Rendering complete.
- Too many points; sampled preview active.
- Invalid field combination.
- Fitting failed or model not suitable.
- 3D surface cannot be generated from collinear or insufficient points.
- 3D surface has duplicate coordinates.
- 3D surface has missing grid cells or an irregular grid.
- 3D surface recommendation available and awaiting user confirmation.
- 3D surface rendering unavailable with a recoverable explanation.

### 7.5 Workspace Step 4 — Customize and Export

#### Figure controls

- Figure title and subtitle.
- X- and Y-axis titles.
- Axis units.
- Interface language independent from figure-text language.
- Font family and font size.
- Line width, marker size, and line style.
- Color palettes and grayscale-safe preview.
- Legend visibility and position.
- Grid and background options.
- Single-column, double-column, A4, and custom-size presets.
- Units: millimeters, centimeters, and inches.

#### Export controls

- PNG.
- SVG.
- PDF.
- 300 DPI.
- 600 DPI.
- Transparent background where supported.
- Black-and-white print preview.
- Export the complete cleaned dataset separately.

Paid export entitlements are intentionally not finalized in this PRD. Show all supported output capabilities in the prototype without adding pricing or checkout UI.

#### Required states

- Export options ready.
- Generating file.
- Export complete with file summary.
- Export failed with retry action.
- Browser download blocked.
- Figure contains elements that may not reproduce well in grayscale.

### 7.6 History

#### Modules

- Local and Cloud segmented control where both are available.
- Search.
- Simple filters for updated date and chart type.
- Project preview cards or a compact list.
- Continue editing, duplicate, export, share, and delete actions.
- Local storage usage and cleanup entry.

#### Required states

- Loading.
- No local projects.
- No cloud projects.
- Offline with local projects available.
- Project file unavailable or damaged.
- Deleting with confirmation.
- Cleanup complete.

### 7.7 Shared Chart

#### Modules

- Responsive interactive chart.
- Title and experiment description.
- Last-updated timestamp.
- Creator-controlled download actions.
- Optional “Copy to my account” action for signed-in users.

#### Required states

- Valid share link.
- Download enabled.
- Download disabled.
- Link expired.
- Link disabled by creator.
- Shared project unavailable.
- Chart loading on a narrow mobile connection.

### 7.8 Settings and Help

#### Settings modules

- Interface language: English or Simplified Chinese.
- Default figure-text language.
- Default font, dimensions, units, and DPI.
- Privacy and data-retention explanation.
- Local project storage usage.
- Clear one project or all local history.
- Cloud storage management for signed-in users.

#### Help modules

- First-chart walkthrough.
- Sample data.
- Supported file formats.
- Explanation of missing values, outliers, standard error, confidence intervals, and fitting.
- Troubleshooting.

### 7.9 Authentication

- Email input.
- Send one-time code.
- Six-digit code entry.
- Resend timer.
- Clear explanation of why login is needed before a cloud save or share action.
- Return the user to the interrupted task after successful authentication.
- Do not require login for Windows local history.

## 8. Responsive Strategy

### Desktop

- Full upload, cleaning, chart editing, fitting, multi-panel layout, and export workflow.
- Primary design target: 1440 × 1024.
- Must remain usable from 1024 px wide.

### Tablet

- Support upload, guided cleaning, standard chart creation, viewing, and export.
- Collapse the right inspector into a drawer.
- Use a compact stepper.

### Mobile

- Primary design target: 390 × 844.
- Support login, history, shared-chart viewing, zooming, downloading, sharing, and simple title or color edits.
- Allow small-file upload and recommended-chart generation.
- Do not attempt row-by-row cleaning, complex multi-panel layout, or advanced 3D configuration.
- For unsupported complex tasks, preserve work and show: “Continue on desktop for advanced editing.”

## 9. Visual Design System Direction

### Visual keywords

- Professional scientific.
- Friendly and reassuring.
- Publication ready.
- Calm.
- Precise.
- Lightweight.

### Color direction

| Role | Suggested color |
| --- | --- |
| Application background | `#F6F8FB` |
| Primary surface | `#FFFFFF` |
| Primary action | `#2563EB` |
| Scientific accent | `#0F766E` |
| Primary text | `#172033` |
| Secondary text | `#667085` |
| Border | `#D8DEE8` |
| Success | `#2E7D5B` |
| Warning | `#B7791F` |
| Error | `#C43D4B` |

Use color to communicate meaning, but never rely on color alone. Data-quality indicators must also use icons, labels, or patterns.

### Typography

- Interface: Inter.
- Chinese interface fallback: Noto Sans SC.
- User-selectable chart fonts should include Arial and Times New Roman.
- Prefer sentence case.
- Use plain language before statistical terminology.

### Layout and components

- Use an 8 px spacing system.
- Use moderate corner radii, approximately 8–12 px.
- Use very subtle shadows and clear borders.
- Use a paper-like white chart canvas.
- Keep forms aligned and dense enough for scientific work without appearing intimidating.
- Prefer segmented controls, select fields, sliders with numeric input, data tables, step indicators, status chips, and contextual help.
- Use skeletons for layout loading and progress indicators for real processing.

### Accessibility

- Target WCAG 2.2 AA contrast and keyboard usability.
- Provide visible focus states.
- Associate every field with a label.
- Do not communicate anomalies only through red coloring.
- Make chart palettes distinguishable for common color-vision deficiencies.
- Allow keyboard access to table issues and chart settings.

## 10. Language and Content Rules

- English is the default interface language.
- Simplified Chinese can be selected at any time.
- Switching the interface language must not automatically translate or replace user-entered figure text.
- Figure title, axis labels, legend text, and export language are controlled independently.
- Error messages must state what happened, whether data is safe, and what the user can do next.
- Avoid unexplained terms such as imputation, interpolation, heteroscedasticity, and confidence band.
- When technical terms are necessary, provide a one-sentence explanation or tooltip.

## 11. Prototype Flows Required in Figma

Create connected high-fidelity prototype screens for these flows:

### Flow A — First chart

1. Empty Home screen.
2. Drag an Excel file into the upload area.
3. Show workbook parsing and progress.
4. Confirm worksheet, header, columns, and units.
5. Show the data-quality result.
6. Accept a recommended line chart.
7. Customize title, units, and color.
8. Export a 300 DPI PNG.

### Flow B — Review suspicious data

1. Open the Inspect and Clean step with several issue types.
2. Select a suspicious point.
3. Read the plain-language reason.
4. Choose to exclude it from the chart without changing the source.
5. Review the before-and-after impact.
6. Undo the decision.

### Flow C — Publication figure

1. Create a fitted curve with equation and R².
2. Select a 95% confidence interval.
3. Add standard-error bars.
4. Arrange four panels.
5. Select a double-column size preset.
6. Preview grayscale output.
7. Open the PNG/SVG/PDF export dialog.

### Flow D — Cloud save and share

1. A guest finishes a chart.
2. The user selects Save to Cloud.
3. Show email one-time-code authentication.
4. Return to the project after login.
5. Save the project.
6. Create a read-only interactive share link.
7. Explicitly enable or disable downloads.

### Flow E — Mobile shared view

1. Open a shared chart at 390 px width.
2. Inspect and zoom the chart.
3. View experiment description.
4. Download when the creator allows it.
5. Handle an expired link gracefully.

## 12. Prototype Deliverables

The Figma prototype should include at minimum:

1. Desktop Home, empty and returning-user variants.
2. Upload and processing states.
3. Import configuration screen.
4. Data-quality workspace with suspicious data selected.
5. Chart editor with a line chart and fitting controls.
6. Four-panel publication layout.
7. Export dialog.
8. History screen.
9. Settings and local-storage cleanup screen.
10. Email one-time-code authentication flow.
11. Desktop shared-chart screen.
12. Mobile shared-chart screen.

Use reusable components and consistent design rules across every generated screen. Generate state variants rather than redesigning the shell for each state.

## 13. V2.0 Non-Goals

- Windows, macOS, or Linux desktop installers in the website-first release.
- Dark mode.
- Google or Microsoft login.
- Real-time multi-user editing.
- Complex joins across multiple uploaded files.
- A code or scripting editor.
- A general-purpose business-intelligence dashboard.
- Automatic modification of suspicious source data.
- A plugin marketplace.
- Live laboratory-instrument ingestion.
- Pricing, subscription checkout, or billing management.

## 14. Product Acceptance Criteria for the Prototype

The prototype is ready for product review when:

- A beginner can identify how to upload a file without instruction.
- The complete upload-to-export path is visible and connected.
- Destructive cleaning is never presented as automatic.
- Every long-running operation has a loading or progress state.
- Empty, error, success, offline, and expired-session states are represented where relevant.
- The chart remains the visual focus during editing.
- Local, temporary cloud, and saved cloud states cannot be confused.
- Desktop and mobile responsibilities are visibly different.
- English is the default, and the language control is easy to find.
- The visual design feels scientific, friendly, and suitable for publication work.

### V2.1 acceptance additions

- A complete `x,y,z` grid is recognized as a surface candidate without silently changing the
  user's chart choice.
- The surface editor makes X/Y/Z roles explicit and shows grid dimensions, usable-point count,
  duplicates, missing cells, and collinearity before rendering.
- A flattened regular grid does not receive false row-order sudden-change findings; a real
  injected discontinuity remains visible and explainable.
- The same surface `ChartSpec` drives the interactive preview and PNG/SVG/PDF export. A browser
  regression verifies the 3D axes, `surfacePoints`, ECharts-GL readiness, camera controls, and
  the absence of a fallback `y over x` line chart.
- Beginner guidance explains why a surface, heatmap, scatter, or line chart is recommended and
  lets the user change that choice.
- V2.1 scientific results disclose assumptions, sample size, exclusions, and limitations; a
  planned method is never presented as implemented evidence.
- BOM and non-ASCII filename regressions pass before the V2.1 release is called compatible.

## 15. Open Decisions After Prototype Review

These decisions must not block initial Figma prototyping:

- Final approval and vector redraw of the generated LabViz logo concept.
- Final cloud plan limits and pricing.
- Which export and automation capabilities belong to a paid plan.
- Exact cloud storage quota for free accounts.
- Exact sharing-link lifetime for signed-in users.
- Final Windows packaging technology for the deferred desktop release.

## 16. Recommended Figma Prototyping Sequence

Use this document as the product source of truth, but build and review the prototype in focused passes rather than generating every screen in one unreviewed operation.

### Pass 1 — Visual direction and shell

- Infer the information architecture and low-fidelity layout directly from this PRD; no hand-drawn wireframe is required.
- Generate the desktop Home screen.
- Generate two visual variations using the same information architecture.
- Select one visual direction before continuing.
- Establish the top bar, workspace navigation, typography, colors, spacing, controls, and chart-canvas treatment.

### Pass 2 — Core desktop workflow

- Reuse the selected shell.
- Use the already attached `assets/labviz-synthetic-table.png` as the visual and content reference for the imported-data preview. It contains synthetic data only.
- Generate Import Data, Inspect and Clean, Create Chart, and Customize and Export.
- Connect Flow A and Flow B.
- Do not redesign navigation or component styling between steps.

### Pass 3 — Scientific figure workflow

- Add fitted-curve, error-bar, confidence-interval, dual-axis, and four-panel states.
- Connect Flow C.
- Check that the chart remains larger and more visually important than the controls.

### Pass 4 — Cloud and account states

- Generate email one-time-code authentication, saved-cloud status, sharing controls, and the Shared Chart screen.
- Connect Flow D.
- Do not invent pricing, checkout, team administration, or additional login providers.

### Pass 5 — Responsive and edge states

- Generate the mobile Shared Chart flow at 390 × 844.
- Add the required empty, loading, error, offline, expired-session, and Large File Mode variants.
- Connect Flow E.

After the visual direction is approved, create a separate frontend-friendly `DESIGN.md` containing the final reusable design-system rules from Figma. Keep product requirements in `prd.md`; do not replace this document with visual tokens alone.

## 17. Figma Implementation Prompts and Review Gates

Use `prd.md`, `assets/labviz-logo.png`, and `assets/labviz-synthetic-table.png` as the source materials for the Figma file. No hand-drawn wireframe is needed: infer and propose the layout from the PRD. Apply the prompts below one pass at a time and review each pass before continuing.

### Prompt 1 — Independent structure exploration and Home direction

```text
I have attached three files: prd.md, labviz-logo.png, and labviz-synthetic-table.png. Read prd.md completely and treat it as the product source of truth for LabViz V2.0. Use labviz-logo.png as the application brand without changing its text or symbol. The table image contains synthetic data and is a later workspace reference, not Home screen content.

No hand-drawn wireframe or existing product screenshot is provided. Independently infer the information architecture and propose the clearest layout for beginner experimenters from the PRD. Do not copy another scientific application or turn the product into a generic BI dashboard.

Work on Pass 1 only. First establish a simple low-fidelity structure for the desktop Home screen and reusable workspace shell. Then turn that structure into two high-fidelity Home screen variations at 1440 × 1024 using the same information architecture. Do not generate the full application yet.

The product is for experimenters who do not know Python. The primary action must be immediately obvious: upload an experimental Excel or CSV file and begin creating a publication-ready chart. Keep secondary choices visually quiet and explain technical concepts in plain language.

Both variations must feel professional scientific, friendly, calm, and publication ready. Use a high-quality light theme, an off-white application background, white surfaces, laboratory blue as the primary action color, restrained teal accents, clear borders, Inter and Noto Sans SC, and a paper-like chart treatment. Avoid dark mode, gradients, glassmorphism, generic BI dashboards, excessive cards, and futuristic styling.

Include the English default interface, a visible Chinese language switch, Local/Cloud context, privacy reassurance, supported formats, a sample-data action, and a restrained recent-project area. Generate reusable shell components. Briefly explain the layout decisions and the important differences between the two variations, then recommend one direction for Pass 2.
```

### Prompt 2 — Core workspace

```text
Continue from the approved LabViz Home direction. Reuse exactly the same design tokens, top bar, typography, spacing, buttons, inputs, status chips, and navigation patterns. Do not redesign the shell. Now use the already attached labviz-synthetic-table.png as a reference for the imported-data preview and its missing-value and suspicious-value states; do not treat it as real experimental data or reproduce it as a static image inside the interface.

Work on Pass 2 from prd.md. Create the connected desktop workflow for Import Data, Inspect and Clean, Create Chart, and Customize and Export. Use a four-step left navigation, a large central data-or-chart canvas, a contextual right inspector, and a compact bottom data-status bar.

Show the Import screen after an Excel file has been parsed, including sheet selection, header detection, column types, unit assignment, and preview. Show the Inspect and Clean screen with several suspicious data points selected, plain-language explanations, suggested actions, before-and-after impact, and undo/redo. Suspicious data must never appear to be deleted automatically.

Connect the upload-to-first-chart flow and the suspicious-data review flow. Include realistic loading, empty, success, and recoverable error states as component variants rather than redesigning the page for every state.
```

### Prompt 3 — Scientific chart workflow

```text
Continue the approved LabViz design system and workspace shell without changing the visual direction. Work on Pass 3 from prd.md.

Create the high-fidelity Create Chart and Customize and Export states for a publication figure. The chart canvas must remain the visual focus and be larger than the control panels.

Show a fitted curve with its equation and R², standard-error bars, a 95% confidence interval selector with 90%, 95%, and 99% options, dual-axis controls, and a four-panel figure layout. Include linear, polynomial up to order three, exponential, logarithmic, and power fitting choices through progressive disclosure.

Create an export dialog with PNG, SVG, and PDF; 300 and 600 DPI; single-column, double-column, A4, and custom dimensions; millimeters, centimeters, and inches; font, line width, units, legend position, and grayscale preview. Do not add pricing or checkout UI.
```

### Prompt 4 — Authentication, cloud save, and sharing

```text
Continue the approved LabViz design system. Work on Pass 4 from prd.md.

Design the interruption-safe cloud save flow for a guest who has completed a chart. Explain why sign-in is required, request an email address, show a six-digit one-time-code screen with resend timing, and return the user to the same project after authentication.

Then design saved-cloud status, sharing controls, and a responsive read-only Shared Chart page. Downloads must be disabled by default and explicitly enabled by the creator. Never expose the original uploaded file. Clearly distinguish Local, Temporary Cloud, and Saved Cloud states.

Do not add passwords, Google login, Microsoft login, pricing, billing, team administration, or real-time collaboration.
```

### Prompt 5 — Mobile and edge states

```text
Continue the approved LabViz design system. Work on Pass 5 from prd.md.

Create a mobile Shared Chart experience at 390 × 844. Support chart viewing, zooming, experiment description, allowed downloads, sharing, and simple title or color edits. Also show an expired-link state and a clear continue-on-desktop message for advanced cleaning, four-panel editing, and advanced 3D configuration.

Create reusable variants for empty upload, file validation, real processing progress, unsupported file, password-protected workbook, no numeric columns, sampled chart preview, Large File Mode, offline state, expired guest session, export failure, and successful export. Preserve the same component language and do not redesign the application for each state.
```
