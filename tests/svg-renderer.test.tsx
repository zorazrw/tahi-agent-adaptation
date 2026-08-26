import { JSDOM } from "jsdom";
import React from "react";
import { act, render } from "@testing-library/react";
import { afterEach, describe, expect, test } from "bun:test";
import { SvgRenderer } from "../src/ui/components/file-renderers/SvgRenderer";
import type { PreviewSaveChrome } from "../src/ui/components/file-renderers/types";

const dom = new JSDOM("<!doctype html><html><body></body></html>", { url: "http://localhost" });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  navigator: dom.window.navigator,
  HTMLElement: dom.window.HTMLElement,
  HTMLIFrameElement: dom.window.HTMLIFrameElement,
  MessageEvent: dom.window.MessageEvent,
});

function sendEditorMessage(iframe: HTMLIFrameElement, type: string, channel: string, extra = {}) {
  window.dispatchEvent(new MessageEvent("message", {
    source: iframe.contentWindow,
    data: { type, channel, ...extra },
  }));
}

afterEach(() => {
  document.body.replaceChildren();
});

function requireChrome(getChrome: () => PreviewSaveChrome | null): PreviewSaveChrome {
  const chrome = getChrome();
  if (!chrome) throw new Error("Save chrome was not registered");
  return chrome;
}

describe("SvgRenderer", () => {
  test("enables Save after a GUI change and writes the original workspace file", async () => {
    const initialSvg = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" /></svg>';
    const changedSvg = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="20" /></svg>';
    const writes: unknown[][] = [];
    let chrome: PreviewSaveChrome | null = null;
    let reloads = 0;

    Object.assign(window, {
      electron: {
        writeFile: async (...args: unknown[]) => {
          writes.push(args);
          return { success: true };
        },
      },
    });

    const view = render(
      <SvgRenderer
        data={{ kind: "svg", content: initialSvg }}
        filePath="artifacts/real-name.svg"
        cwd="/workspace"
        sessionId="session-1"
        onReload={() => { reloads += 1; }}
        onTextSaveChromeChange={(next) => { chrome = next; }}
      />
    );
    const iframe = view.getByTitle("Interactive SVG editor") as HTMLIFrameElement;
    const channel = decodeURIComponent(new URL(iframe.src).hash.slice(1));

    expect(requireChrome(() => chrome).disabled).toBe(true);
    await act(async () => {
      sendEditorMessage(iframe, "agent-cowork:svg-ready", channel);
      sendEditorMessage(iframe, "agent-cowork:svg-loaded", channel, { svg: initialSvg });
    });
    expect(requireChrome(() => chrome).disabled).toBe(true);

    await act(async () => {
      sendEditorMessage(iframe, "agent-cowork:svg-changed", channel, { svg: changedSvg });
    });
    expect(requireChrome(() => chrome).disabled).toBe(false);

    const agentSvg = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="8" /></svg>';
    view.rerender(
      <SvgRenderer
        data={{ kind: "svg", content: agentSvg }}
        filePath="artifacts/real-name.svg"
        cwd="/workspace"
        sessionId="session-1"
        onReload={() => { reloads += 1; }}
        onTextSaveChromeChange={(next) => { chrome = next; }}
      />
    );
    expect(requireChrome(() => chrome).disabled).toBe(false);
    expect(view.getByText(/changed on disk while the canvas has unsaved edits/).textContent)
      .toContain("Save to overwrite it");

    await act(async () => {
      sendEditorMessage(iframe, "agent-cowork:svg-changed", channel, { svg: changedSvg });
    });
    expect(view.getByText(/changed on disk while the canvas has unsaved edits/).textContent)
      .toContain("Save to overwrite it");

    await act(async () => {
      requireChrome(() => chrome).save();
      await Promise.resolve();
    });

    expect(writes).toEqual([[
      "artifacts/real-name.svg",
      "/workspace",
      changedSvg,
      "session-1",
    ]]);
    expect(reloads).toBe(1);
  });
});
