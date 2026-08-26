import { createRequire } from "node:module";
import { describe, expect, test } from "bun:test";

const require = createRequire(import.meta.url);
const installSvgEditBridge = require("../scripts/svg-edit-bridge.cjs") as (options: BridgeOptions) => {
  editor: FakeEditor;
  dispose: () => void;
};

type Listener = (event: { source: unknown; data?: Record<string, unknown> }) => void;
type Deferred = () => void;

type BridgeOptions = {
  EditorCtor: typeof FakeEditor;
  container: object;
  childWindow: FakeChildWindow;
  parentWindow: FakeParentWindow;
  channel: string;
  defer: (callback: Deferred) => void;
};

class FakeSvgCanvas {
  events: Record<string, ((win: unknown, elements: unknown) => void) | undefined> = {};
  svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" /></svg>';

  bind(event: string, callback: (win: unknown, elements: unknown) => void) {
    const previous = this.events[event];
    this.events[event] = callback;
    return previous;
  }

  call(event: string, elements?: unknown) {
    this.events[event]?.({}, elements);
  }

  getSvgString() {
    return this.svg;
  }

  setSvgString(svg: string) {
    this.svg = svg;
    return true;
  }
}

class FakeEditor {
  static latest: FakeEditor;
  svgCanvas = new FakeSvgCanvas();
  topPanel = { updateTitle: (title: string) => { this.title = title; } };
  title = "untitled.svg";
  readyCallback: (() => void) | null = null;
  internalChangedCalls = 0;
  internalSourceChangedCalls = 0;

  constructor(container: object) {
    void container;
    FakeEditor.latest = this;
  }

  setConfig(config: object) {
    void config;
  }
  init() {}
  ready(callback: () => void) {
    this.readyCallback = callback;
  }

  finishStartup() {
    this.readyCallback?.();
    this.svgCanvas.bind("changed", () => {
      this.internalChangedCalls += 1;
    });
    this.svgCanvas.bind("sourcechanged", () => {
      this.internalSourceChangedCalls += 1;
    });
  }
}

class FakeParentWindow {
  messages: Array<Record<string, unknown>> = [];
  postMessage(message: Record<string, unknown>, target: string) {
    void target;
    this.messages.push(message);
  }
}

class FakeChildWindow {
  listener: Listener | null = null;
  addEventListener(_event: string, listener: Listener) {
    this.listener = listener;
  }
  removeEventListener(_event: string, listener: Listener) {
    if (this.listener === listener) this.listener = null;
  }
  dispatch(source: unknown, data: Record<string, unknown>) {
    this.listener?.({ source, data });
  }
}

function setup() {
  const deferred: Deferred[] = [];
  const parentWindow = new FakeParentWindow();
  const childWindow = new FakeChildWindow();
  const installed = installSvgEditBridge({
    EditorCtor: FakeEditor,
    container: {},
    childWindow,
    parentWindow,
    channel: "test-channel",
    defer: (callback) => deferred.push(callback),
  });
  const flush = () => {
    while (deferred.length) deferred.shift()?.();
  };
  return { ...installed, parentWindow, childWindow, flush };
}

describe("SVG-Edit bridge", () => {
  test("chains after SVG-Edit installs its single changed callback", () => {
    const { editor, parentWindow, flush } = setup();

    editor.finishStartup();
    expect(parentWindow.messages).toHaveLength(0);
    flush();
    expect(parentWindow.messages[0]?.type).toBe("agent-cowork:svg-ready");

    editor.svgCanvas.svg = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="5" /></svg>';
    editor.svgCanvas.call("changed", []);

    expect(editor.internalChangedCalls).toBe(1);
    expect(parentWindow.messages[1]?.type).toBe("agent-cowork:svg-changed");
    expect(parentWindow.messages[1]?.svg).toContain("<circle");
  });

  test("forwards source-editor changes without breaking SVG-Edit callbacks", () => {
    const { editor, parentWindow, flush } = setup();
    editor.finishStartup();
    flush();
    parentWindow.messages.length = 0;

    editor.svgCanvas.svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>edited source</text></svg>';
    editor.svgCanvas.call("sourcechanged", []);

    expect(editor.internalSourceChangedCalls).toBe(1);
    expect(parentWindow.messages[0]?.type).toBe("agent-cowork:svg-changed");
    expect(parentWindow.messages[0]?.svg).toContain("edited source");
  });

  test("loads source, reports canonical SVG, and displays the real filename", () => {
    const { editor, parentWindow, childWindow, flush } = setup();
    editor.finishStartup();
    flush();
    parentWindow.messages.length = 0;

    const svg = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0" /></svg>';
    childWindow.dispatch(parentWindow, {
      type: "agent-cowork:set-svg",
      channel: "test-channel",
      svg,
      fileName: "diagram.svg",
    });
    flush();

    expect(editor.title).toBe("diagram.svg");
    expect(editor.svgCanvas.getSvgString()).toBe(svg);
    expect(parentWindow.messages[0]?.type).toBe("agent-cowork:svg-loaded");
    expect(parentWindow.messages[0]?.svg).toBe(svg);
  });

  test("ignores messages from the wrong source or channel", () => {
    const { editor, parentWindow, childWindow, flush } = setup();
    editor.finishStartup();
    flush();
    const initial = editor.svgCanvas.getSvgString();

    childWindow.dispatch({}, {
      type: "agent-cowork:set-svg",
      channel: "test-channel",
      svg: "<svg></svg>",
    });
    childWindow.dispatch(parentWindow, {
      type: "agent-cowork:set-svg",
      channel: "wrong-channel",
      svg: "<svg></svg>",
    });

    expect(editor.svgCanvas.getSvgString()).toBe(initial);
  });

  test("dispose removes the iframe message listener", () => {
    const { childWindow, dispose } = setup();
    expect(childWindow.listener === null).toBe(false);
    dispose();
    expect(childWindow.listener).toBe(null);
  });
});
