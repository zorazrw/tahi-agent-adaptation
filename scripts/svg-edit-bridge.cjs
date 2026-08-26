(function exposeBridge(root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory;
  } else {
    root.installAgentCoworkSvgBridge = factory;
  }
})(typeof globalThis === "undefined" ? window : globalThis, function installSvgEditBridge(options) {
  const { EditorCtor, container, childWindow, parentWindow, channel, defer = queueMicrotask } = options;
  const send = (type, extra = {}) => parentWindow.postMessage({ type, channel, ...extra }, "*");
  const editor = new EditorCtor(container);
  let loading = false;

  editor.setConfig({ allowInitialUserOverride: false, extensions: [], noDefaultExtensions: false });
  editor.init();
  editor.ready(() => {
    defer(() => {
      const forwardChange = (eventName) => {
        const previous = editor.svgCanvas.bind(eventName, (win, elements) => {
          previous?.(win, elements);
          if (!loading) {
            send("agent-cowork:svg-changed", { svg: editor.svgCanvas.getSvgString() });
          }
        });
      };
      forwardChange("changed");
      forwardChange("sourcechanged");
      send("agent-cowork:svg-ready");
    });
  });

  const receiveMessage = (event) => {
    if (event.source !== parentWindow || event.data?.channel !== channel) return;
    if (event.data?.type !== "agent-cowork:set-svg" || typeof event.data.svg !== "string") return;

    loading = true;
    const loaded = editor.svgCanvas?.setSvgString(event.data.svg, true);
    if (loaded === false) {
      loading = false;
      send("agent-cowork:svg-error", { error: "SVG-Edit could not load this document." });
      return;
    }
    if (typeof event.data.fileName === "string" && event.data.fileName) {
      editor.topPanel?.updateTitle(event.data.fileName);
    }
    defer(() => {
      send("agent-cowork:svg-loaded", { svg: editor.svgCanvas.getSvgString() });
      loading = false;
    });
  };

  childWindow.addEventListener("message", receiveMessage);
  return {
    editor,
    dispose: () => childWindow.removeEventListener("message", receiveMessage),
  };
});
