import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = resolve(root, "node_modules/svgedit/dist/editor");
const bridgeSource = resolve(root, "scripts/svg-edit-bridge.cjs");
const target = resolve(root, "public/svgedit");

await rm(target, { recursive: true, force: true });
await mkdir(dirname(target), { recursive: true });
await cp(source, target, { recursive: true });
await cp(bridgeSource, resolve(target, "agent-cowork-bridge.js"));
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
  <script src="./agent-cowork-bridge.js"></script>
  <script>
    const EditorCtor = (window.Editor && window.Editor.default) || window.Editor;
    window.installAgentCoworkSvgBridge({
      EditorCtor,
      container: document.getElementById('container'),
      childWindow: window,
      parentWindow: window.parent,
      channel: decodeURIComponent(window.location.hash.slice(1)),
    });
  </script>
</body>
</html>`);
