# XLSX editing in the preview panel — Univer + ExcelJS integration plan

**Status:** Implemented on branch `feature/xlsx-univer-editing` (build-green; verified with a bridge smoke test; **not yet verified inside the running Electron app**). This doc is the design + the remaining-work checklist, written to be self-contained.

**Goal:** Extend the file-preview panel from *read-only* `.xlsx` rendering to **view + edit** `.xlsx`, and **capture the user's edits** as agent-readable signal — all **fully client-side** (no server/Docker), inside the Electron app.

**Chosen solution:** **Univer Core** (editable spreadsheet UI, in the renderer) + **ExcelJS** (`.xlsx` read/write, in the main process), bridged by a neutral JSON model. Univer has a rich action-capture stream and an offline formula engine; ExcelJS is the free writer used here to preserve common cell styling. Both are permissively licensed (Apache-2.0 / MIT).

---

## 1. Why this shape (the constraints that forced it)

1. **The renderer is sandboxed.** `main.ts` creates the `BrowserWindow` with only `preload` set, so Electron defaults apply: `sandbox:true`, `contextIsolation:true`, `nodeIntegration:false`. The renderer has **no Node/Buffer/fs** → **ExcelJS must run in the main process**; the renderer talks to it over the typed IPC bridge (`window.electron.*`).
2. **The save path was UTF-8-string only.** The existing `write-file` IPC hard-rejects non-string content and does `writeFile(resolved, content, "utf8")`. A real `.xlsx` is a binary ZIP → we need a **binary write path** (`write-xlsx`).
3. **Capture is text-diff based.** `recordFileEditAfterPreviewSave` (`src/electron/ipc-handlers.ts:1148`) records a `file_edit` message + a UTF-8 content snapshot, then `gatherHumanFileEditDiffs` (`src/electron/libs/file-edit-diffs.ts:23`) **text-diffs** that snapshot to refine the agent's verifiers. A binary blob does not diff meaningfully → we record a **textual (TSV) rendering** of the workbook so the existing pipeline yields cell-level deltas.
4. **No-server is non-negotiable for an inline panel** → eliminates Univer Pro/Server and ONLYOFFICE. Univer **Core** (no Pro) is fine; its only gap is native `.xlsx` I/O, which ExcelJS fills.

### Architecture

```
.xlsx on disk
   │  preview-file (IPC)
   ▼
MAIN (ExcelJS)  ── fileToModel ──►  XlsxModel (plain JSON, crosses contextBridge)
   ▲                                   │  createWorkbook
   │  write-xlsx (IPC)                 ▼
MAIN (ExcelJS)  ◄── modelToBytes ──  RENDERER (Univer)  ── edit + onCommandExecuted (capture)
   │                                   ▲
   ├─ writeFile(bytes)  → disk         └─ workbookDataToModel ── on Save
   └─ modelToText → recordFileEditAfterPreviewSave  (TSV → existing text-diff → agent)
```

- **MAIN owns all byte I/O** and never imports `@univerjs` (keeps the main bundle lean, no enum coupling).
- **RENDERER owns the Univer mapping** (the Univer enums live in `univer-model.ts`).
- They communicate only via the neutral `XlsxModel`.

---

## 2. The neutral model (the IPC contract)

`src/lib/xlsx-model.ts` (shared; `src/lib/` is already imported by both main and renderer). Plain, JSON-serializable. 0-based `r`/`c` to match Univer's cell matrix.

- `XlsxModel { fileName, sheets: XlsxSheet[] }`
- `XlsxSheet { name, rowCount, colCount, cells: XlsxCell[], merges, freeze?, colWidths? }`
- `XlsxCell { r, c, v?, f?(no leading "="), numFmt?, bold?, italic?, underline?, size?, fontName?, color?(#RRGGBB), fill?, wrap?, hAlign?, vAlign?, border? }`

Borders carry ExcelJS style names (`thin`/`medium`/…) so they round-trip: the renderer maps them to Univer's `BorderStyleTypes` for display, and main maps them back to ExcelJS on write.

---

## 3. Files

### New
| File | Role |
|---|---|
| `src/lib/xlsx-model.ts` | Neutral model types (the IPC payload). |
| `src/electron/libs/exceljs-xlsx.ts` | `fileToModel` (read, full styles) · `modelToBytes` (write, full styles incl. borders/merges/freeze) · `modelToText` (TSV capture). |
| `src/ui/components/file-renderers/univer-model.ts` | `modelToWorkbookData` / `workbookDataToModel` (XlsxModel ↔ Univer `IWorkbookData`; owns the enum mapping). |
| `src/ui/components/file-renderers/UniverSheet.tsx` | Lazy Univer editor. Mounts the workbook, captures edits via `onCommandExecuted` (→ dirty), exposes `getModel()` via ref. |
| `src/ui/components/file-renderers/XlsHtmlRenderer.tsx` | Legacy `.xls` read-only HTML (former `SpreadsheetRenderer` behavior). |

### Modified
| File | Change |
|---|---|
| `src/electron/main.ts` | `preview-file`: `.xlsx` → `fileToModel` → `{kind:"xlsx", model}` (falls back to read-only HTML `{kind:"xls"}` on parse failure); `.xls` → `{kind:"xls", sheets}`. **New `write-xlsx` handler**: `modelToBytes` → binary `writeFile` + `recordFileEditAfterPreviewSave(sid, path, modelToText(model))`. |
| `src/electron/preload.cts` | Exposes `writeXlsx(filePath, cwd, model, sessionId)`. |
| `types.d.ts` | `PreviewFileResult`: `xlsx → {model}`, add `xls → {sheets}`; `EventPayloadMapping["write-xlsx"]`; `Window.electron.writeXlsx`. |
| `src/ui/components/file-renderers/SpreadsheetRenderer.tsx` | Rewritten: lazy wrapper around `UniverSheet`; manages dirty/saving/error; surfaces `PreviewSaveChrome` (Save button + ⌘S via `FilePreview`); on save calls `window.electron.writeXlsx`. Keyed by `filePath` to remount per file. |
| `src/ui/components/file-renderers/index.ts` | Registry: `xlsx → SpreadsheetRenderer` (editable), `xls → XlsHtmlRenderer` (read-only). |
| `src/ui/components/FilePreview.tsx` | Local `PreviewFileResult` updated to match (`xlsx → {model}`, add `xls`). |
| `src/ui/components/file-renderers/html-preview-edit.ts` | Pre-existing `globalThis.setTimeout`→`window.setTimeout` (the `@types/node` bump from the new deps exposed a `Timeout` vs `number` typing issue). |

### Dependencies added
`@univerjs/core`, `@univerjs/preset-sheets-core`, `@univerjs/themes` (`^0.24`), `exceljs` (`^4.4`), `rxjs` (`^7.8`).

---

## 4. Action capture (the "capture those actions" requirement)

- The renderer marks the doc dirty on real **user** edit commands via `univerAPI.onCommandExecuted`, filtering for `sheet.command.set-range-values` and the structural/format/style command ids.
  - **Gotcha (fixed):** do NOT listen for the `sheet.mutation.set-range-values` *mutation* — Univer's formula engine fires it during load/recompute, which falsely marks the doc dirty. (Caught in the live demo.)
- On Save, the renderer hands the current `XlsxModel` to main. Main writes the styled `.xlsx` **and** records `modelToText(model)` (a deterministic per-sheet TSV, formulas shown as `=…`) through the existing `recordFileEditAfterPreviewSave`.
- Result: `gatherHumanFileEditDiffs` text-diffs consecutive TSV snapshots → the agent sees cell-level deltas like `Budget!B4: 12000 → 13000`, not an opaque blob. **No new capture infrastructure required.**

A DEV-only handle `window.__xlsxEditorApi` is set in `UniverSheet` (guarded by `import.meta.env.DEV`, stripped from production) — used for headless smoke-tests and a future hook for **agent-driven edits**.

---

## 5. Fidelity (verified, not assumed)

Round-tripped a generated workbook fixture (values, number formats, styling, and frozen panes) through the ExcelJS bridge.

| Property | Round-trips? |
|---|---|
| Values, formulas (+ Univer-recomputed cached results) | ✅ |
| Number formats (`"$"#,##0`, `0.0%`) | ✅ |
| Merged cells, frozen panes | ✅ |
| **Cell styling — fills, fonts, bold/italic, color, borders** | ✅ (via ExcelJS; community SheetJS would **drop** these) |
| Charts, pivot tables, sparklines, data validation, conditional formats | ❌ (ExcelJS re-serializes the whole workbook through its own model; unmodeled features are dropped) |

> The styling win is the whole reason for ExcelJS over the SheetJS path the app already ships. Caveat: a full re-serialize drops advanced objects ExcelJS doesn't model — see "ZIP-patch" under Open Decisions for the higher-fidelity option.

**Verification:** `bun run build` passes. A throwaway `bun` smoke test generated an `.xlsx`, read it into `XlsxModel`, edited a cell, wrote bytes back out, re-read them, and confirmed the TSV capture included the edit.

**Build:** `bun run transpile:electron` ✅, `tsc -b` ✅, `vite build` ✅ (Univer is isolated in a ~5.4 MB **lazy chunk** — loads only when an `.xlsx` is previewed, not in the main bundle). ESLint clean on new files.

---

## 6. The integration point in the app UI

`FilePreview` is rendered in `src/ui/App.tsx` (~line 660). Its `filePath` comes from the **selected workflow node's output files**:
```
getPreviewFileForNode(node?.outputFiles)   // App.tsx ~673, keyed off selectedNodeId
```
So an `.xlsx` reaches the editable panel when a workflow node lists one in `outputFiles` and that node is selected. No UI change is needed for the editor to appear — it's wired through the renderer registry (`xlsx → SpreadsheetRenderer`). `FilePreview` already renders the `PreviewSaveChrome` Save button + ⌘S handler, and already has an "Open in Finder" action.

---

## 7. Remaining work / checklist

- [ ] **Live in-app verification** (the main gap). Boot the app (`bun run dev`), get an `.xlsx` into the preview panel, edit a cell, Save, and confirm: file updated on disk + a `file_edit` row recorded with the TSV. Path to get a file previewed: a session whose selected workflow node has an `.xlsx` in `outputFiles` (App.tsx:673). Options to set up without a full agent run: (a) run a quick agent task that writes an `.xlsx`; (b) seed a session row in `app.getPath("userData")/sessions.db` with a node whose `outputFiles` points at a test `.xlsx` on disk; (c) add a temporary dev affordance to preview an arbitrary file.
- [ ] **Decide save/export UX in-app** (see Open Decisions).
- [ ] **Decide what ships vs stays dev-only**: keep the DEV `__xlsxEditorApi` hook? If bridge tests are needed long term, add committed tests under the repo's test structure instead of relying on local experiment harnesses.
- [ ] **Bundle hygiene**: Univer pulls ~40 locale chunks; prune to en-US. Confirm the lazy chunk doesn't regress cold-preview latency in the packaged app.
- [ ] **Edge cases to exercise**: empty/huge workbooks; many sheets; cells with rich text / dates / errors; insert/delete row+col then Save (full-snapshot rewrite handles shifts — confirm); a styled-but-empty cell; `.xls` still renders read-only; non-parseable `.xlsx` falls back to read-only HTML.
- [ ] **Number/date edge cases**: ExcelJS reads dates as display text in the model (no serial); confirm acceptable, or add date-serial handling.
- [ ] **electron-builder packaging**: confirm `exceljs` (and its deps) are bundled for the main process in `dist:*` builds; confirm Univer assets are included in `dist-react`.
- [ ] **Tests in CI**: wire the two bridge tests into the test runner.

---

## 8. Open decisions (need a call)

1. **Save vs Download/Export in-app.** The implemented behavior is **in-place Save** — write the edited bytes back to the previewed file's path (this is what the agent reads; matches a desktop app). The demo also had a browser **Download**, but in-app that's largely redundant with the existing **"Open in Finder"**. Options: (a) in-place Save only [current], (b) add an explicit "Export As…"/Download, (c) both.
2. **Styling-perfect saves.** Current full-rewrite via ExcelJS preserves fonts/fills/borders/merges/freeze but drops charts/pivots/etc. it doesn't model. If users round-trip richly-decorated source files, switch the writer to a **changed-cell ZIP-patch** (keep the original `xl/*.xml`, rewrite only edited cells) — uniquely enabled because Univer's `onCommandExecuted` tells us exactly which cells changed. Higher fidelity, more plumbing. Defer unless needed.
3. **`.xls` (legacy binary).** Currently read-only HTML (ExcelJS has no `.xls` reader). Leave as-is, or convert via SheetJS into the same `XlsxModel` to make `.xls` editable too.
4. **Capture granularity.** Today the agent gets cell deltas from the TSV diff. If we want explicit structured deltas (`{sheet,addr,old,new,op}`), also pass the renderer's captured command stream to a richer `recordFileEditAfterPreviewSave` (or a new `spreadsheet_edit` action). Optional.

---

## 9. Risk & rollback

- **Additive & guarded.** New files + a handful of edits. `.xlsx` parse failure falls back to the old read-only HTML path; `.xls` is unchanged. Other preview types untouched.
- **Rollback** = restore `SpreadsheetRenderer.tsx` (read-only HTML), revert the `preview-file` `.xlsx` branch + remove `write-xlsx`, drop the new files and deps.
- **Main-process dependency** (`exceljs`) is the only new runtime dep in main; everything Univer is renderer-only and lazy.

---

## 10. How to run / see it today

- **Build:** `bun run build`
- **Smoke test:** generate an `.xlsx` with ExcelJS, run it through `fileToModel` → edit the neutral model → `modelToBytes` → `fileToModel`, and confirm `modelToText` includes the edit.

---

## Appendix — key code references

- `src/electron/main.ts` — `preview-file` handler (xlsx → model) + `write-xlsx` handler; `createWindow` `webPreferences` (~line 234, sandbox defaults).
- `src/electron/ipc-handlers.ts:1148` — `recordFileEditAfterPreviewSave`.
- `src/electron/libs/file-edit-diffs.ts:23` — `gatherHumanFileEditDiffs` (text diff).
- `src/electron/libs/message-state-snapshot.ts:256` — `buildExportEnvironmentSnapshotWithPreviewWrittenFile`.
- `src/ui/App.tsx:~660-690` — `FilePreview` mount; `filePath` from `getPreviewFileForNode(node.outputFiles)`.
- `src/ui/components/FilePreview.tsx` — `PreviewSaveChrome`, ⌘S, renderer routing.
- `src/ui/components/file-renderers/index.ts` — renderer registry + `EditableRendererProps`/`PreviewSaveChrome`.
