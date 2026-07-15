"""Per-epoch token log-probability visualization for offline DPO.

Reuses the per-token policy log probabilities that are already materialized
inside the DPO loss closure (``chosen_logprob_seqs`` / ``rejected_logprob_seqs``
in :mod:`scripts.weight.train.run_dpo`).  No extra inference is performed here;
this module only decodes tokens and renders HTML/JSONL.

Each per-token logprob tensor is aligned 1:1 with the datum's
``loss_fn_inputs["target_tokens"]`` and ``loss_fn_inputs["weights"]``.  Tokens
with ``weight > 0`` are the actually-trained region (the artifact ``write()``
content); tokens with ``weight == 0`` are shared prompt/context.

Both *chosen* and *rejected* trajectories are rendered in the HTML (toggle each
on/off).  A per-epoch page (``epoch_XXX.html``) shows the trajectories captured
that epoch, and a cross-epoch page (``compare.html``) lets you place the same
data point's chosen (and rejected) trajectory from two different epochs side by
side, so you can scrutinize how the model's per-token log probabilities shift
over training.  Data points are matched across epochs via a stable ``pair_key``
derived from the (identical, across epochs) chosen token content and shared by
the paired rejected sample, so epoch shuffling does not break the pairing.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # Only needed for type hints; avoid hard runtime deps so the
    import tinker  # HTML regeneration CLI can run without torch/tinker installed.
    import torch


def _decode_token(tokenizer: Any, token_id: int) -> str:
    """Best-effort decode of a single token id to its text piece."""
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:
        text = ""
    return str(text)


def _stable_key(datum: tinker.Datum, target_tokens: Sequence[int]) -> str:
    """Stable identity for a data point, consistent across epochs.

    Uses the datum's ``model_input`` tokens (prompt + chosen response) when
    available, falling back to the target tokens.  For a given chosen
    trajectory the content is identical across epochs, so this key matches the
    same data point regardless of epoch shuffling.
    """
    ids: list[int]
    try:
        ids = [int(x) for x in datum.model_input.to_ints()]
    except Exception:
        ids = [int(x) for x in target_tokens]
    return hashlib.sha1(json.dumps(ids).encode("utf-8")).hexdigest()[:12]


def _build_one(
    datum: tinker.Datum,
    logprob_seq: torch.Tensor,
    tokenizer: Any,
    label: str,
    pair_index: int,
) -> dict[str, Any]:
    target_tokens = list(datum.loss_fn_inputs["target_tokens"].data)
    weights = list(datum.loss_fn_inputs["weights"].data)
    logprobs = [float(x) for x in logprob_seq.detach().float().cpu().tolist()]

    n = min(len(target_tokens), len(weights), len(logprobs))

    tokens: list[dict[str, Any]] = []
    trained_logprob_sum = 0.0
    trained_count = 0
    for i in range(n):
        tok = int(target_tokens[i])
        weight = float(weights[i])
        logprob = float(logprobs[i])
        tokens.append(
            {
                "text": _decode_token(tokenizer, tok),
                "token_id": tok,
                "logprob": logprob,
                "weight": weight,
            }
        )
        if weight > 0.0:
            trained_logprob_sum += logprob
            trained_count += 1

    return {
        "label": label,
        "pair_index": pair_index,
        "key": _stable_key(datum, target_tokens),
        "tokens": tokens,
        "trained_logprob_sum": trained_logprob_sum,
        "trained_logprob_mean": (
            trained_logprob_sum / trained_count if trained_count else 0.0
        ),
        "trained_token_count": trained_count,
    }


def build_samples(
    chosen_data: Sequence[tinker.Datum],
    rejected_data: Sequence[tinker.Datum],
    chosen_lp_seqs: Sequence[torch.Tensor],
    rejected_lp_seqs: Sequence[torch.Tensor],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Build per-sample token/logprob records for chosen and rejected data.

    Samples are interleaved per pair: ``chosen_0, rejected_0, chosen_1, ...``.
    Both sides are rendered in the HTML. Each pair shares a ``pair_key`` (the
    chosen side's stable key), so a data point's chosen AND rejected samples can
    be matched across epochs even though the rejected content changes per epoch.
    """
    samples: list[dict[str, Any]] = []
    n_pairs = min(
        len(chosen_data), len(rejected_data), len(chosen_lp_seqs), len(rejected_lp_seqs)
    )
    for i in range(n_pairs):
        chosen = _build_one(chosen_data[i], chosen_lp_seqs[i], tokenizer, "chosen", i)
        rejected = _build_one(
            rejected_data[i], rejected_lp_seqs[i], tokenizer, "rejected", i
        )
        # The task identity is the (across-epochs stable) chosen key; share it
        # with the rejected side so both can be paired across epochs.
        pair_key = chosen["key"]
        chosen["pair_key"] = pair_key
        rejected["pair_key"] = pair_key
        samples.append(chosen)
        samples.append(rejected)
    return samples


def write_jsonl(samples: Sequence[dict[str, Any]], epoch: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            record = {"epoch": epoch, **sample}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Shared front-end assets (injected as .format() arguments so their braces are
# never processed by str.format).                                             #
# --------------------------------------------------------------------------- #

_SHARED_CSS = """
  :root { color-scheme: dark light; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 24px; background: #14161a; color: #e6e6e6;
  }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #9aa0a6; font-size: 13px; margin-bottom: 20px; }
  .sub a { color: #7ea6ff; }
  .controls {
    position: sticky; top: 0; background: #14161a; padding: 12px 0;
    border-bottom: 1px solid #2a2d33; margin-bottom: 16px; z-index: 10;
    display: flex; flex-wrap: wrap; align-items: center; gap: 16px;
  }
  .controls label { font-size: 13px; cursor: pointer; }
  .controls select {
    background: #1b1e24; color: #e6e6e6; border: 1px solid #2a2d33;
    border-radius: 4px; padding: 3px 6px; font-size: 13px;
  }
  .legend { display: inline-flex; align-items: center; gap: 8px; font-size: 12px; color: #9aa0a6; }
  .legend .bar {
    width: 160px; height: 12px; border-radius: 3px;
    background: linear-gradient(to right,
      hsl(0,70%,45%), hsl(30,70%,45%), hsl(60,70%,42%), hsl(90,70%,38%), hsl(120,65%,38%));
  }
  .sample {
    border: 1px solid #2a2d33; border-radius: 8px; margin-bottom: 16px;
    padding: 12px 14px; background: #1b1e24;
  }
  .sample-header { font-size: 13px; margin-bottom: 8px; }
  .badge {
    display: inline-block; padding: 1px 8px; border-radius: 10px;
    font-weight: 600; font-size: 12px; margin-right: 8px;
  }
  .badge.chosen { background: #12351f; color: #7ee29b; }
  .badge.epoch { background: #1d2740; color: #9ec1ff; }
  .meta { color: #9aa0a6; }
  .delta.up { color: #7ee29b; }
  .delta.down { color: #f08a9d; }
  .body { line-height: 2.1; font-size: 14px; word-break: break-word; white-space: pre-wrap; }
  .body.values { line-height: 1.5; }
  .tok { border-radius: 3px; padding: 1px 0; }
  .tok.trained { padding: 1px 1px; }
  .tok.context { color: #6b7078; background: transparent !important; }
  .body.values .tok.trained {
    display: inline-flex; flex-direction: column; align-items: center;
    padding: 1px 3px; margin: 1px 1px; vertical-align: bottom; line-height: 1.15;
  }
  .body.values .tok .tt { white-space: pre-wrap; }
  .lp { font-size: 9px; opacity: 0.85; font-variant-numeric: tabular-nums; margin-top: 1px; }
  .cmp-row {
    display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 16px;
  }
  .cmp-row .sample { margin-bottom: 0; }
  .cmp-key { color: #6b7078; font-size: 11px; font-family: ui-monospace, monospace; }
  .empty { color: #9aa0a6; font-style: italic; }
  .dp { border-top: 2px solid #2a2d33; padding-top: 12px; margin-bottom: 24px; }
  .dp-head { font-size: 13px; color: #c7ccd1; margin-bottom: 8px; }
  .badge.rejected { background: #3a1720; color: #f08a9d; }
"""

# Token-rendering helpers shared by both pages.
_HELPERS_JS = """
const LOGPROB_MIN = -8.0; // clamp floor for color scaling
function colorFor(logprob) {
  let norm = (logprob - LOGPROB_MIN) / (0.0 - LOGPROB_MIN);
  norm = Math.max(0, Math.min(1, norm));
  const hue = norm * 120; // 0 red -> 120 green
  return `hsl(${hue}, 70%, 30%)`;
}
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function groupWords(tokens) {
  const words = [];
  let cur = null;
  for (const t of tokens) {
    const startsWord = cur === null || /^[\\s]/.test(t.text) || /^[\\u2581\\u0120]/.test(t.text);
    if (startsWord) {
      cur = { text: t.text, logprob: t.logprob, weight: t.weight };
      words.push(cur);
    } else {
      cur.text += t.text;
      cur.logprob += t.logprob;
      cur.weight = Math.max(cur.weight, t.weight);
    }
  }
  return words;
}
function renderToken(t, showContext, showValues) {
  const trained = t.weight > 0;
  if (!trained && !showContext) return "";
  const cls = ["tok", trained ? "trained" : "context"];
  const display = esc(t.text).replace(/\\n/g, "<br>");
  const prob = Math.exp(t.logprob);
  let style = "";
  if (trained) style = `background:${colorFor(t.logprob)};`;
  const title = `logprob=${t.logprob.toFixed(3)} p=${prob.toFixed(4)} weight=${t.weight}`;
  if (trained && showValues) {
    return `<span class="${cls.join(" ")}" style="${style}" title="${title}">`
      + `<span class="tt">${display}</span>`
      + `<span class="lp">${t.logprob.toFixed(2)}</span></span>`;
  }
  return `<span class="${cls.join(" ")}" style="${style}" title="${title}">${display}</span>`;
}
function renderBody(tokens, wordMode, showContext, showValues) {
  const items = wordMode ? groupWords(tokens) : tokens;
  const body = items.map(t => renderToken(t, showContext, showValues)).join("");
  const cls = showValues ? "body values" : "body";
  return `<div class="${cls}">${body}</div>`;
}
"""

_EPOCH_JS = _HELPERS_JS + """
const ALL = JSON.parse(document.getElementById("data").textContent);
function render() {
  const wordMode = document.getElementById("wordToggle").checked;
  const showContext = document.getElementById("contextToggle").checked;
  const showValues = document.getElementById("valueToggle").checked;
  const showChosen = document.getElementById("showChosen").checked;
  const showRejected = document.getElementById("showRejected").checked;
  const root = document.getElementById("root");
  // Interleave chosen/rejected by pair so paired samples stay adjacent.
  const samples = ALL.filter(s =>
    (s.label === "chosen" && showChosen) || (s.label === "rejected" && showRejected)
  );
  if (!samples.length) {
    root.innerHTML = '<div class="empty">Nothing to show. Enable chosen and/or rejected.</div>';
    return;
  }
  root.innerHTML = samples.map(s => {
    const body = renderBody(s.tokens, wordMode, showContext, showValues);
    return `<div class="sample">
      <div class="sample-header">
        <span class="badge ${s.label}">${s.label}</span>
        <span class="meta">pair #${s.pair_index} &middot; trained tokens: ${s.trained_token_count} &middot; sum logprob: ${s.trained_logprob_sum.toFixed(3)} &middot; mean logprob: ${s.trained_logprob_mean.toFixed(3)}</span>
        <span class="cmp-key"> &middot; ${s.pair_key || s.key || ""}</span>
      </div>
      ${body}
    </div>`;
  }).join("");
}
["valueToggle","wordToggle","contextToggle","showChosen","showRejected"].forEach(
  id => document.getElementById(id).addEventListener("change", render)
);
render();
"""

_COMPARE_JS = _HELPERS_JS + """
const DATA = JSON.parse(document.getElementById("data").textContent);
// DATA = { epochs: [...], samples: [ {epoch, label, pair_key, pair_index, tokens, ...} ] }
// Index: BY[epoch][label][pair_key] = sample
const BY = {};
for (const s of DATA.samples) {
  const e = (BY[s.epoch] = BY[s.epoch] || {});
  const l = (e[s.label] = e[s.label] || {});
  l[s.pair_key] = s;
}
function fillSelect(sel, epochs, defaultVal) {
  sel.innerHTML = epochs.map(e => `<option value="${e}">epoch ${e}</option>`).join("");
  sel.value = String(defaultVal);
}
function sampleCard(s, epoch, label, wordMode, showContext, showValues) {
  const body = renderBody(s.tokens, wordMode, showContext, showValues);
  return `<div class="sample">
    <div class="sample-header">
      <span class="badge ${label}">${label}</span>
      <span class="badge epoch">epoch ${epoch}</span>
      <span class="meta">sum logprob: ${s.trained_logprob_sum.toFixed(3)} &middot; mean: ${s.trained_logprob_mean.toFixed(3)} &middot; tokens: ${s.trained_token_count}</span>
    </div>
    ${body}
  </div>`;
}
function labelRow(label, a, b, mapA, mapB, k, wordMode, showContext, showValues) {
  const sA = (mapA[label] || {})[k];
  const sB = (mapB[label] || {})[k];
  if (!sA || !sB) return "";
  const delta = sB.trained_logprob_sum - sA.trained_logprob_sum;
  const dcls = delta >= 0 ? "delta up" : "delta down";
  const sign = delta >= 0 ? "+" : "";
  return `<div class="sub">${label} &middot; <span class="${dcls}">&Delta; sum logprob ${sign}${delta.toFixed(3)}</span></div>
    <div class="cmp-row">
      ${sampleCard(sA, a, label, wordMode, showContext, showValues)}
      ${sampleCard(sB, b, label, wordMode, showContext, showValues)}
    </div>`;
}
function render() {
  const a = document.getElementById("epochA").value;
  const b = document.getElementById("epochB").value;
  const wordMode = document.getElementById("wordToggle").checked;
  const showContext = document.getElementById("contextToggle").checked;
  const showValues = document.getElementById("valueToggle").checked;
  const showChosen = document.getElementById("showChosen").checked;
  const showRejected = document.getElementById("showRejected").checked;
  const mapA = BY[a] || {};
  const mapB = BY[b] || {};
  const wanted = [];
  if (showChosen) wanted.push("chosen");
  if (showRejected) wanted.push("rejected");
  // Collect all data-point keys, ordered by chosen pair_index in epoch A.
  const keyOrder = {};
  for (const label of ["chosen", "rejected"]) {
    for (const k of Object.keys(mapA[label] || {})) {
      if (!(k in keyOrder)) keyOrder[k] = (mapA[label][k] || {}).pair_index ?? 0;
    }
  }
  const keys = Object.keys(keyOrder).sort((k1, k2) => keyOrder[k1] - keyOrder[k2]);
  const root = document.getElementById("root");
  const blocks = [];
  for (const k of keys) {
    const rows = wanted.map(label =>
      labelRow(label, a, b, mapA, mapB, k, wordMode, showContext, showValues)
    ).filter(Boolean);
    if (!rows.length) continue;
    blocks.push(`<div class="dp"><div class="dp-head">data point <span class="cmp-key">${k}</span></div>${rows.join("")}</div>`);
  }
  if (!blocks.length) {
    root.innerHTML = '<div class="empty">No data points are present in both selected epochs for the chosen filters. Pick two epochs with overlapping data points, or enable chosen/rejected.</div>';
    return;
  }
  root.innerHTML = blocks.join("");
}
(function init() {
  const epochs = DATA.epochs.slice();
  const defA = epochs[0];
  const defB = epochs[epochs.length - 1];
  fillSelect(document.getElementById("epochA"), epochs, defA);
  fillSelect(document.getElementById("epochB"), epochs, defB);
  ["epochA","epochB","valueToggle","wordToggle","contextToggle","showChosen","showRejected"].forEach(
    id => document.getElementById(id).addEventListener("change", render)
  );
  render();
})();
"""

_EPOCH_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Token Log-Probabilities — Epoch {epoch}</title>
<style>{shared_css}</style>
</head>
<body>
<h1>Token Log-Probabilities — Epoch {epoch}</h1>
<div class="sub">Chosen and rejected trajectories from the epoch's first batch, colored by policy per-token log probability (reused from the DPO loss, no extra inference). The number under each token is its exact logprob. See <a href="compare.html">compare.html</a> to place the same data point across two epochs side by side.</div>
<div class="controls">
  <label><input type="checkbox" id="showChosen" checked> Chosen</label>
  <label><input type="checkbox" id="showRejected" checked> Rejected</label>
  <label><input type="checkbox" id="valueToggle" checked> Show logprob values</label>
  <label><input type="checkbox" id="wordToggle"> Group subwords into words</label>
  <label><input type="checkbox" id="contextToggle" checked> Show context tokens</label>
  <span class="legend">low logprob <span class="bar"></span> high logprob</span>
</div>
<div id="root"></div>
<script id="data" type="application/json">{data_json}</script>
<script>{page_js}</script>
</body>
</html>
"""

_COMPARE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trajectory Comparison Across Epochs</title>
<style>{shared_css}</style>
</head>
<body>
<h1>Trajectory Comparison Across Epochs</h1>
<div class="sub">Pick two epochs to place each data point's chosen and/or rejected trajectory side by side. Only data points captured in both epochs are shown. Colors and numbers are policy per-token log probabilities. (Rejected content differs per epoch since it is resampled each rollout.)</div>
<div class="controls">
  <label>Epoch A <select id="epochA"></select></label>
  <label>Epoch B <select id="epochB"></select></label>
  <label><input type="checkbox" id="showChosen" checked> Chosen</label>
  <label><input type="checkbox" id="showRejected" checked> Rejected</label>
  <label><input type="checkbox" id="valueToggle" checked> Show logprob values</label>
  <label><input type="checkbox" id="wordToggle"> Group subwords into words</label>
  <label><input type="checkbox" id="contextToggle" checked> Show context tokens</label>
  <span class="legend">low logprob <span class="bar"></span> high logprob</span>
</div>
<div id="root"></div>
<script id="data" type="application/json">{data_json}</script>
<script>{page_js}</script>
</body>
</html>
"""


def _embed_json(obj: Any) -> str:
    """Serialize JSON for safe embedding inside a <script> block."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def render_html(samples: Sequence[dict[str, Any]], epoch: int, out_path: Path) -> None:
    """Write the per-epoch page (chosen and rejected trajectories)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = _EPOCH_TEMPLATE.format(
        epoch=epoch,
        shared_css=_SHARED_CSS,
        data_json=_embed_json(list(samples)),
        page_js=_EPOCH_JS,
    )
    out_path.write_text(doc, encoding="utf-8")


def render_epoch(
    chosen_data: Sequence[tinker.Datum],
    rejected_data: Sequence[tinker.Datum],
    chosen_lp_seqs: Sequence[torch.Tensor],
    rejected_lp_seqs: Sequence[torch.Tensor],
    tokenizer: Any,
    epoch: int,
    out_dir: Path,
) -> Path:
    """Build samples and write ``epoch_XXX.html``/``.jsonl`` plus ``compare.html``.

    Returns the path to the written per-epoch HTML file.
    """
    samples = build_samples(
        chosen_data, rejected_data, chosen_lp_seqs, rejected_lp_seqs, tokenizer
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"epoch_{epoch:03d}.html"
    jsonl_path = out_dir / f"epoch_{epoch:03d}.jsonl"
    render_html(samples, epoch, html_path)
    write_jsonl(samples, epoch, jsonl_path)
    _write_compare(out_dir)
    _write_index(out_dir)
    return html_path


def _fallback_key(rec: dict[str, Any]) -> str:
    token_ids = [t.get("token_id") for t in rec.get("tokens", [])]
    return hashlib.sha1(json.dumps(token_ids).encode("utf-8")).hexdigest()[:12]


def _read_epoch_records(jsonl_path: Path) -> list[dict[str, Any]]:
    """Read one ``epoch_*.jsonl`` and return records with ``pair_key`` ensured.

    ``pair_key`` is the shared per-pair (task) identity. When absent (older
    JSONL), it is reconstructed within the epoch: the chosen record's stable key
    for a given ``pair_index`` is assigned to both sides of that pair.
    """
    records: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Map pair_index -> chosen stable key (for reconstructing pair_key).
    chosen_key_by_pair: dict[Any, str] = {}
    for rec in records:
        if rec.get("label") == "chosen":
            chosen_key_by_pair[rec.get("pair_index")] = rec.get("key") or _fallback_key(rec)

    for rec in records:
        if not rec.get("pair_key"):
            pi = rec.get("pair_index")
            rec["pair_key"] = (
                chosen_key_by_pair.get(pi)
                or rec.get("key")
                or _fallback_key(rec)
            )
        if not rec.get("key"):
            rec["key"] = _fallback_key(rec)
    return records


def _load_all_across_epochs(out_dir: Path) -> tuple[list[int], list[dict[str, Any]]]:
    """Read every ``epoch_*.jsonl`` and collect chosen + rejected samples."""
    epochs: set[int] = set()
    samples: list[dict[str, Any]] = []
    for jsonl in sorted(out_dir.glob("epoch_*.jsonl")):
        for rec in _read_epoch_records(jsonl):
            epoch = int(rec.get("epoch", 0))
            epochs.add(epoch)
            samples.append(
                {
                    "epoch": epoch,
                    "label": rec.get("label", "chosen"),
                    "key": rec.get("key"),
                    "pair_key": rec.get("pair_key"),
                    "pair_index": rec.get("pair_index", 0),
                    "tokens": rec.get("tokens", []),
                    "trained_logprob_sum": rec.get("trained_logprob_sum", 0.0),
                    "trained_logprob_mean": rec.get("trained_logprob_mean", 0.0),
                    "trained_token_count": rec.get("trained_token_count", 0),
                }
            )
    return sorted(epochs), samples


def _write_compare(out_dir: Path) -> None:
    """Write ``compare.html`` embedding chosen + rejected samples from all epochs."""
    epochs, samples = _load_all_across_epochs(out_dir)
    doc = _COMPARE_TEMPLATE.format(
        shared_css=_SHARED_CSS,
        data_json=_embed_json({"epochs": epochs, "samples": samples}),
        page_js=_COMPARE_JS,
    )
    (out_dir / "compare.html").write_text(doc, encoding="utf-8")


def rebuild_from_jsonl(out_dir: Path) -> list[Path]:
    """Regenerate all HTML pages from existing ``epoch_*.jsonl`` files.

    Useful for updating the renderer (e.g. to include rejected trajectories)
    without re-running training. Returns the list of rewritten epoch HTML paths.
    """
    out_dir = Path(out_dir)
    written: list[Path] = []
    for jsonl in sorted(out_dir.glob("epoch_*.jsonl")):
        records = _read_epoch_records(jsonl)
        if not records:
            continue
        epoch = int(records[0].get("epoch", 0))
        html_path = out_dir / f"{jsonl.stem}.html"
        render_html(records, epoch, html_path)
        written.append(html_path)
    _write_compare(out_dir)
    _write_index(out_dir)
    return written


def _write_index(out_dir: Path) -> None:
    """Maintain a simple index.html linking to all pages."""
    epochs = sorted(out_dir.glob("epoch_*.html"))
    links = "\n".join(
        f'<li><a href="{p.name}">{html.escape(p.stem)}</a></li>' for p in epochs
    )
    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Token Log-Prob Visualizations</title>"
        "<style>body{font-family:-apple-system,sans-serif;background:#14161a;"
        "color:#e6e6e6;padding:24px;}a{color:#7ea6ff;}</style></head>"
        "<body><h1>Token Log-Prob Visualizations</h1>"
        '<p><a href="compare.html">Compare trajectories across epochs &rarr;</a></p>'
        f"<ul>{links}</ul></body></html>"
    )
    (out_dir / "index.html").write_text(doc, encoding="utf-8")


__all__ = [
    "build_samples",
    "write_jsonl",
    "render_html",
    "render_epoch",
    "rebuild_from_jsonl",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate token log-prob HTML from existing epoch_*.jsonl."
    )
    parser.add_argument(
        "out_dir",
        help="logprob_viz directory containing epoch_*.jsonl files.",
    )
    args = parser.parse_args()
    paths = rebuild_from_jsonl(Path(args.out_dir))
    for p in paths:
        print(f"wrote {p}")
    print(f"wrote {Path(args.out_dir) / 'compare.html'}")
    print(f"wrote {Path(args.out_dir) / 'index.html'}")
