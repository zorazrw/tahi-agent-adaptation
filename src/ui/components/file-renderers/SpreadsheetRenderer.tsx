import { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react";
import type { XlsxModel } from "../../../lib/xlsx-model";
import type { EditableRendererProps } from "./index";
import type { UniverSheetHandle } from "./UniverSheet";

// Heavy Univer bundle loads only when an .xlsx is actually previewed.
const UniverSheet = lazy(() => import("./UniverSheet"));

type Props = { data: { kind: "xlsx"; model: XlsxModel } } & EditableRendererProps;

export function SpreadsheetRenderer({
  data,
  filePath,
  cwd,
  sessionId,
  onTextSaveChromeChange,
}: Props) {
  const sheetRef = useRef<UniverSheetHandle>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // New file -> reset edit state.
  useEffect(() => {
    setDirty(false);
    setSaving(false);
    setError(null);
  }, [filePath]);

  const handleSave = useCallback(async () => {
    if (!filePath || saving) return;
    const model = sheetRef.current?.getModel();
    if (!model) {
      setError("Could not read the spreadsheet state");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const res = await window.electron.writeXlsx(filePath, cwd ?? undefined, model, sessionId ?? undefined);
      if (res.success) setDirty(false);
      else setError(res.error ?? "Failed to save");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }, [filePath, cwd, sessionId, saving]);

  // Surface the preview-panel Save affordance through shared chrome.
  useEffect(() => {
    if (!onTextSaveChromeChange) return;
    if (!dirty && !saving && !error) {
      onTextSaveChromeChange(null);
      return;
    }
    onTextSaveChromeChange({
      save: () => void handleSave(),
      disabled: !dirty || saving,
      saving,
      error,
    });
    return () => onTextSaveChromeChange(null);
  }, [dirty, saving, error, handleSave, onTextSaveChromeChange]);

  if (!data?.model?.sheets?.length) {
    return <p className="text-sm text-muted-foreground p-2">No sheets found</p>;
  }

  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
      <Suspense
        fallback={
          <div className="flex items-center gap-2 text-sm text-muted-foreground p-2">
            <span>Loading spreadsheet editor...</span>
          </div>
        }
      >
        <UniverSheet
          key={filePath ?? "xlsx"}
          ref={sheetRef}
          model={data.model}
          onEdit={() => setDirty(true)}
          onError={(m) => setError(m)}
        />
      </Suspense>
    </div>
  );
}
