/**
 * Univer editable spreadsheet (lazy-loaded; pulls in the heavy Univer bundle
 * only when an .xlsx is previewed).
 *
 * - Renders an XlsxModel via Univer's offline canvas engine (real formulas).
 * - Captures edits through Univer's command bus (onCommandExecuted) and marks
 *   the document dirty so the preview-panel Save affordance appears.
 * - Exposes getModel() so the wrapper can pull the current state back to a
 *   neutral XlsxModel and hand it to MAIN (ExcelJS) for a styled .xlsx write.
 */
import {
  forwardRef,
  useEffect,
  useId,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

import { LogLevel, LocaleType, merge, Univer } from "@univerjs/core";
import { FUniver } from "@univerjs/core/lib/facade";
import { defaultTheme } from "@univerjs/themes";
import { UniverSheetsCorePreset } from "@univerjs/preset-sheets-core";
import UniverPresetSheetsCoreEnUS from "@univerjs/preset-sheets-core/locales/en-US";
import "@univerjs/preset-sheets-core/lib/index.css";

import type { XlsxModel } from "../../../lib/xlsx-model";
import { modelToWorkbookData, workbookDataToModel } from "./univer-model";

type UniverAPI = FUniver;

function createSheetsUniver(container: string) {
  const univer = new Univer({
    logLevel: LogLevel.WARN,
    locale: LocaleType.EN_US,
    locales: { [LocaleType.EN_US]: merge({}, UniverPresetSheetsCoreEnUS) },
    theme: defaultTheme,
  });

  for (const plugin of UniverSheetsCorePreset({ container }).plugins) {
    const [pluginCtor, options] = Array.isArray(plugin) ? [plugin[0], plugin[1]] : [plugin, undefined];
    univer.registerPlugin(pluginCtor, options);
  }

  return { univer, univerAPI: FUniver.newAPI(univer) };
}

export type UniverSheetHandle = {
  /** Current workbook state as a neutral model (for the ExcelJS write in MAIN). */
  getModel: () => XlsxModel | null;
};

/**
 * Command ids that represent a real USER document edit (mark dirty).
 * IMPORTANT: listen for the user COMMAND `sheet.command.set-range-values`, NOT the
 * `sheet.mutation.set-range-values` mutation: the formula engine fires that
 * mutation during initial load/recompute, which would falsely mark the doc dirty.
 */
const EDIT_COMMANDS = new Set<string>([
  "sheet.command.set-range-values",
  "sheet.command.insert-row",
  "sheet.command.insert-row-before",
  "sheet.command.insert-row-after",
  "sheet.command.remove-row",
  "sheet.command.insert-col",
  "sheet.command.remove-col",
  "sheet.command.move-rows",
  "sheet.command.move-cols",
  "sheet.command.delete-range-move-up",
  "sheet.command.delete-range-move-left",
  "sheet.command.insert-range-move-down",
  "sheet.command.insert-range-move-right",
  "sheet.command.add-worksheet-merge",
  "sheet.command.add-worksheet-merge-all",
  "sheet.command.remove-worksheet-merge",
  "sheet.command.set-bold",
  "sheet.command.set-italic",
  "sheet.command.set-underline",
  "sheet.command.set-background-color",
  "sheet.command.set-text-color",
  "sheet.command.set-border-basic",
  "sheet.command.set-border",
  "sheet.command.set-text-wrap",
  "sheet.command.set-horizontal-text-align",
  "sheet.command.set-vertical-text-align",
  "sheet.command.numfmt.set.numfmt",
  "sheet.command.set-style",
  "sheet.command.set-worksheet-col-width",
  "sheet.command.set-worksheet-row-height",
]);

type Props = {
  model: XlsxModel;
  /** Fires on every captured edit (used to mark the document dirty). */
  onEdit?: () => void;
  onError?: (message: string) => void;
};

export const UniverSheet = forwardRef<UniverSheetHandle, Props>(function UniverSheet(
  { model, onEdit, onError },
  ref
) {
  const containerId = useId().replace(/[^a-zA-Z0-9_-]/g, "_");
  const apiRef = useRef<UniverAPI | null>(null);
  const onEditRef = useRef(onEdit);
  onEditRef.current = onEdit;
  const [ready, setReady] = useState(false);

  useImperativeHandle(
    ref,
    () => ({
      getModel: () => {
        const api = apiRef.current;
        const wb = api?.getActiveWorkbook();
        if (!wb) return null;
        try {
          return workbookDataToModel(wb.getSnapshot(), model.fileName);
        } catch {
          return null;
        }
      },
    }),
    [model.fileName]
  );

  useEffect(() => {
    let disposed = false;
    let instance: { dispose: () => void } | null = null;

    const host = document.getElementById(containerId);
    if (host) host.innerHTML = ""; // clear any stale DOM (StrictMode re-mount safety)

    try {
      const { univer, univerAPI } = createSheetsUniver(containerId);
      if (disposed) {
        univer.dispose();
        return;
      }
      instance = univer;
      apiRef.current = univerAPI;

      univerAPI.createWorkbook(modelToWorkbookData(model));

      // Dev-only handle for headless smoke-tests / future programmatic (agent) edits.
      // Stripped from production builds via the import.meta.env.DEV guard.
      if (import.meta.env.DEV) {
        (window as unknown as Record<string, unknown>).__xlsxEditorApi = univerAPI;
      }

      univerAPI.onCommandExecuted((command: { id: string }) => {
        if (EDIT_COMMANDS.has(command.id)) onEditRef.current?.();
      });

      setReady(true);
    } catch (e) {
      onError?.(e instanceof Error ? e.message : String(e));
    }

    return () => {
      disposed = true;
      try {
        instance?.dispose();
      } catch {
        /* noop */
      }
      apiRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [containerId]);

  return (
    <div
      style={{
        flex: 1,
        minHeight: 0,
        position: "relative",
        opacity: ready ? 1 : 0.6,
      }}
    >
      <div id={containerId} style={{ position: "absolute", inset: 0 }} />
    </div>
  );
});

export default UniverSheet;
