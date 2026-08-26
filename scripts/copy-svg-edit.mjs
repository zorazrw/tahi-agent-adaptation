import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = resolve(root, "node_modules/svgedit/dist/editor");
const target = resolve(root, "public/svgedit");

await rm(target, { recursive: true, force: true });
await mkdir(dirname(target), { recursive: true });
await cp(source, target, { recursive: true });
await Promise.all([
  rm(resolve(target, "tests"), { recursive: true, force: true }),
  rm(resolve(target, "Editor.js"), { force: true }),
  rm(resolve(target, "Editor.js.map"), { force: true }),
  rm(resolve(target, "iife-Editor.js.map"), { force: true }),
]);
await writeFile(resolve(target, "index.html"), `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
  <link href="./svgedit.css" rel="stylesheet">
  <title>Agent Cowork SVG Editor</title>
</head>
<body style="margin:0;overflow:hidden">
  <div id="container" style="width:100%;height:100vh"></div>
  <script src="./iife-Editor.js"></script>
  <script>
    const channel = decodeURIComponent(window.location.hash.slice(1));
    const send = (type, extra = {}) => window.parent.postMessage({ type, channel, ...extra }, '*');
    const EditorCtor = (window.Editor && window.Editor.default) || window.Editor;
    const editor = new EditorCtor(document.getElementById('container'));
    let loading = false;

    editor.setConfig({ allowInitialUserOverride: false, extensions: [], noDefaultExtensions: false });
    editor.init();
    editor.ready(() => {
      editor.svgCanvas.bind('changed', () => {
        if (!loading) send('agent-cowork:svg-changed', { svg: editor.svgCanvas.getSvgString() });
      });
      send('agent-cowork:svg-ready');
    });

    window.addEventListener('message', (event) => {
      if (event.source !== window.parent || event.data?.channel !== channel) return;
      if (event.data?.type !== 'agent-cowork:set-svg' || typeof event.data.svg !== 'string') return;
      loading = true;
      editor.svgCanvas?.setSvgString(event.data.svg, true);
      queueMicrotask(() => {
        send('agent-cowork:svg-loaded', { svg: editor.svgCanvas.getSvgString() });
        loading = false;
      });
    });
  </script>
</body>
</html>`);
