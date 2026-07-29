# LabViz V2.0 Project Plan

**Status:** Frontend V2.0 alpha implemented; production backend architecture not yet approved
**Product source of truth:** [`prd.md`](prd.md)
**Prototype source of truth:** [LabViz V2.0 — Product Prototype](https://www.figma.com/design/xKdEwynLhAj2dyiqeEw58n)

## 1. Delivery Strategy

V2.0 is **website-first**. The first implementation is a responsive web application; a Windows desktop shell is deferred until the web workflow and Python processing API are stable.

The initial product path is:

```text
Upload → Inspect → Create chart → Export
```

The first vertical slice must let a user select a supported file, review a bounded data preview and quality summary, configure a line chart, and prepare a publication export without requiring Python knowledge.

## 2. Approved Frontend Stack

| Concern | Decision |
| --- | --- |
| Application framework | Next.js App Router with strict TypeScript |
| Component framework | Material UI Core |
| Data preview | MUI X Data Grid Community only |
| Interactive charts | Apache ECharts with SVG rendering where practical |
| Remote state | TanStack Query when the Python API is connected |
| Workspace state | Zustand; keep large datasets outside the browser store |
| Forms and validation | React Hook Form and Zod when forms require schema validation |
| Internationalization | next-intl; English default, Simplified Chinese optional |
| Unit tests | Vitest |
| Browser flow tests | Playwright desktop, mobile, mock-contract, screenshot-regression, WCAG, and opt-in live-service flows |

Do not introduce a custom primitive component library, Redux, micro-frontends, a monorepo package layer, Storybook, or desktop-runtime code unless a demonstrated requirement justifies it.

MUI X Pro and Premium features are not allowed in the open-core baseline. A paid third-party UI license must never become necessary to run the core application.

## 3. Frontend Structure

```text
V2.0/web/
├── src/app/                 Next.js routes and route-level composition
├── src/features/            Product capabilities and their local UI/state
├── src/components/layout/   Application shell and workflow layout
├── src/components/common/   Reusable LabViz-specific composites
├── src/domain/              Serializable project and chart contracts
├── src/lib/                 Cross-cutting adapters and utilities
└── src/theme/               MUI theme and design tokens
```

Folders are created only when they contain real code. A future desktop application may extract shared packages only after reuse is proven.

## 4. Module Boundaries

### Processing

The future Python processing API is the source of truth for parsing, missing-value detection, anomaly explanations, cleaning, statistics, fitting, and publication export. The repository currently includes a FastAPI contract/reference service so the frontend can be tested end to end; this does not pre-approve the production backend framework or deployment architecture. The frontend may validate obvious file constraints and render previews, but must not implement a competing scientific calculation path.

### Chart configuration

UI controls edit a serializable, versioned `ChartSpec`. Interactive preview, saved projects, shared views, and backend export must consume the same specification. React components and library-specific ECharts objects must not be persisted in it.

### Project configuration

A versioned `ProjectSpec` stores source metadata, user-approved cleaning decisions, chart configuration, and export preferences. Original binary data and large row arrays do not belong in client state.

### Storage and authentication

Pages call adapters rather than browser storage or cloud services directly. This keeps a future local SQLite implementation replaceable without changing product screens.

## 5. Frontend API Contract

The frontend consumes a versioned JSON contract at `/api/v1`. Set
`NEXT_PUBLIC_LABVIZ_API_URL` when the Python API is hosted on another origin.
Every response includes `apiVersion: "v1"` and is validated with Zod before it
enters workspace state.

| Method and path | Frontend purpose |
| --- | --- |
| `POST /projects` | Upload a real file and create a temporary processing project. |
| `POST /samples/thermal-response/projects` | Create the explicit server-provided sample project. |
| `GET /projects/{projectId}` | Reopen a saved or temporary project. |
| `GET /jobs/{jobId}` | Poll real parsing and profiling progress. |
| `GET /projects/{projectId}/preview` | Retrieve bounded typed rows and sampling metadata. |
| `GET /projects/{projectId}/quality` | Retrieve explained quality findings and row references. |
| `PATCH /projects/{projectId}/cleaning-decisions` | Save explicit keep, exclude, or cleaned-copy removal decisions. |
| `PUT /projects/{projectId}/chart` | Save the versioned `ChartSpec` used by preview, sharing, and export. |
| `POST /projects/{projectId}/shares` | Create a read-only link with downloads disabled or enabled explicitly. |
| `GET /projects` | Populate History with saved project summaries. |
| `GET /shares/{token}` | Populate the read-only Shared Chart view and download permissions. |
| `POST /auth/email-code` | Request an email one-time-code challenge. |
| `POST /auth/email-code/verify` | Verify the six-digit code and return the signed-in user. |
| `POST /projects/{projectId}/exports` | Request a publication export from the saved `ChartSpec`. |

The frontend does not fall back to generated rows if the API is unavailable.
It displays an explicit loading, empty, expired, or recoverable error state so
synthetic values cannot be mistaken for processed experiment data.

## 6. Component Policy

Use MUI primitives directly and customize them centrally through the theme. Do not create wrappers that merely rename `Button`, `TextField`, `Select`, or `Dialog`.

Encapsulate LabViz-specific composites such as:

- `AppShell`
- `UploadDropzone`
- `WorkflowStepper`
- `DataPreviewTable`
- `QualityIssuePanel`
- `ChartCanvas`
- `ProcessingProgress`
- `ExportPanel`

## 7. Design Tokens

Figma values are mapped into one MUI theme using semantic roles:

- action and focus colors;
- canvas, surface, border, and text colors;
- success, warning, error, and suspicious-data states;
- typography and bilingual font fallbacks;
- an 8 px spacing scale;
- radii, shadows, breakpoints, and chart palettes.

Feature components must not repeat raw brand colors or invent local spacing scales. Light mode is the only V2.0 implementation target.

## 8. Delivery Phases

### Phase 1 — Frontend skeleton

- Create the Next.js/MUI application and theme.
- Implement Home and the four-step workspace shell.
- Establish the versioned frontend API contract and remove generated preview rows.
- Establish `ChartSpec`, file policy, state, localization, tests, and production build.

### Phase 2 — API-backed vertical slice

**Status:** Completed against the local FastAPI contract/reference service and covered by integration and browser tests on 2026-07-29.

- Upload a real CSV/XLSX file to FastAPI.
- Display parsed metadata, a bounded preview, progress, and quality issues.
- Record an explicit user cleaning decision.
- Render a line chart from the returned preview data.
- Request and download a 300 DPI PNG from the API.

### Phase 3 — Scientific figure tools

**Status:** Frontend and contract/reference-service alpha complete for all items below.

- Add remaining chart types, fitting, error bars, confidence intervals, dual axes, and up to four panels.
- Add SVG/PDF, 600 DPI, journal dimensions, grayscale QA, and complete export states.

### Phase 4 — Cloud product

**Status:** Frontend alpha and local contract/reference-service flows complete
for temporary retention, email-code sessions, saved history, and sharing.
Production backend selection, managed storage, and real SMTP deployment remain.

- Add guest retention, email-code authentication, saved projects, history, and sharing.
- Add mobile shared views and the remaining responsive states.

### Deferred

- Windows packaging, local SQLite, and local Python sidecar.
- macOS/Linux installers, dark mode, external identity providers, billing, and team features.

## 9. Phase 1 Acceptance Criteria

- `npm run dev` starts the web application from `V2.0/web`.
- English is the default and the language control can switch to Simplified Chinese.
- A supported file or explicit server-provided sample can enter the API-backed workspace.
- The user can move through Import, Inspect, Chart, and Export screens.
- The chart is the visual focus and the layout remains usable at 1024 px width.
- Invalid type and cloud files above 50 MB receive clear feedback.
- No screen renders generated rows or claims that unavailable API data was scientifically processed.
- Lint, type checking, unit tests, and the production build all pass.
