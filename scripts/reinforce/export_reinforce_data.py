#!/usr/bin/env python3
"""REINFORCE-style export with LLM-rated verifier rewards."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

import induce  # noqa: E402
import session_export_common as s  # noqa: E402

DEFAULT_VERIFIER_MODEL = "claude-haiku-4-5-20251001"
_ACTION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(([\s\S]*)\)$")
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\n\r]+$")


def _strip_env(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip_env(v) for k, v in x.items() if k != "environment"}
    if isinstance(x, list):
        return [_strip_env(i) for i in x]
    return x


def _session_dict(meta: dict, initial: Any, units: list | None, *, error: str | None = None) -> dict:
    """Assemble session payload; strip env then actor for export."""
    body: dict[str, Any] = {**meta, "initial_message": initial, "learning_units": units or []}
    if error:
        body["error"] = error
    return s.strip_actor(_strip_env(body))


@dataclass
class VerifierScorer:
    """Anthropic verifier: criteria + file snapshots → pass rate in [0, 1]."""

    model: str
    max_tokens: int
    cache: dict[str, float] = field(default_factory=dict)
    debug_prompts: bool = False
    client: Any | None = None
    no_llm_reward: bool = False
    max_retries: int = 1

    def score(self, criteria: list[str], file_blocks: list[tuple[str, str]]) -> float:
        if self.no_llm_reward:
            return 0.5
        if not criteria or not file_blocks or self.client is None:
            return 0.0

        def sha256_key() -> str:
            h = hashlib.sha256()
            for c in criteria:
                h.update(c.encode("utf-8"))
                h.update(b"\0")
            for rel, content in file_blocks:
                h.update(rel.encode("utf-8"))
                h.update(b"\0")
                h.update(content.encode("utf-8"))
                h.update(b"\0")
            return h.hexdigest()

        key = sha256_key()
        if (cached := self.cache.get(key)) is not None:
            return cached

        def trunc(text: str, max_len: int = 14_000) -> str:
            return text if len(text) <= max_len else text[:max_len] + "\n... [truncated]"

        def redact_long_base64_json_strings(text: str) -> str:
            def repl(m: re.Match[str]) -> str:
                raw = m.group(1)
                if len(raw) <= 160:
                    return m.group(0)
                preview = raw[: min(48, len(raw))]
                return f'"content": "[image-bytes-truncated len={len(raw)} preview={preview}...]"'

            return re.sub(r'"content"\s*:\s*"([A-Za-z0-9+/=\n\r]{800,})"', repl, text)

        def is_base64ish(st: str, min_len: int = 256) -> bool:
            t = st.strip()
            return len(t) >= min_len and _BASE64_RE.fullmatch(t) is not None

        def image_base64_for_path(rel: str, text: str) -> tuple[str, str] | None:
            media_type, _ = mimetypes.guess_type(rel)
            if not media_type or not media_type.startswith("image/"):
                return None
            raw = text.strip()
            if is_base64ish(raw):
                return media_type, raw
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                c = parsed.get("content")
                if isinstance(c, str) and is_base64ish(c):
                    return media_type, c.strip()
            return None

        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria))
        intro = "\n".join(
            [
                "You are an automated checker for completed task output files.",
                "Given verifier criteria and current output files, decide whether each criterion is satisfied.",
                'Reply with ONLY a JSON object of this exact shape: {"results":[{"pass":true},{"pass":false},...]}',
                "The results array must have exactly one object per verifier line, in the same order (indices 0 .. n-1).",
                "",
                "Verifier criteria (in order):",
                numbered,
            ]
        )
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": intro},
            {"type": "text", "text": "Output files and contents:"},
        ]
        if not file_blocks:
            blocks.append({"type": "text", "text": "(no output files present)"})
        else:
            for rel, text in file_blocks:
                blocks.append({"type": "text", "text": f"### {rel}"})
                img = image_base64_for_path(rel, text)
                if img is not None:
                    media_type, data = img
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": data},
                        }
                    )
                    blocks.append({"type": "text", "text": "(image content attached above)"})
                else:
                    blocks.append({"type": "text", "text": trunc(redact_long_base64_json_strings(text))})
                blocks.append({"type": "text", "text": "---"})

        if self.debug_prompts:
            lines: list[str] = []
            for i, block in enumerate(blocks):
                bt = block.get("type")
                if bt == "text":
                    lines.append(f"[{i}] text:\n{block.get('text', '')}")
                elif bt == "image":
                    src = block.get("source") if isinstance(block.get("source"), dict) else {}
                    data = str(src.get("data") or "")
                    lines.append(
                        f"[{i}] image: media_type={src.get('media_type')} len={len(data)} preview={data[:48]}..."
                    )
                else:
                    lines.append(f"[{i}] {json.dumps(block, ensure_ascii=False)}")
            print("\n=== verifier_prompt_begin ===", file=sys.stderr)
            print("\n".join(lines), file=sys.stderr)
            print("=== verifier_prompt_end ===\n", file=sys.stderr)

        def json_object_from_model_text(text: str) -> dict[str, Any]:
            fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            raw = (fence.group(1) if fence else text).strip()
            lo, hi = raw.find("{"), raw.rfind("}")
            if lo == -1 or hi == -1 or hi <= lo:
                raise ValueError("no json object")
            return json.loads(raw[lo : hi + 1])

        def parse_pass_flags(text: str, n: int) -> list[bool | None]:
            try:
                obj = json_object_from_model_text(text)
                rows = obj.get("results")
                if not isinstance(rows, list):
                    raise ValueError("no results")
                out: list[bool | None] = [None] * n
                for i in range(min(n, len(rows))):
                    row = rows[i]
                    if isinstance(row, dict) and "pass" in row:
                        out[i] = bool(row["pass"])
                return out
            except (ValueError, json.JSONDecodeError, TypeError, KeyError):
                tokens = re.findall(r'"pass"\s*:\s*(true|false)', text, flags=re.IGNORECASE)
                out2: list[bool | None] = [None] * n
                for i, tok in enumerate(tokens[:n]):
                    out2[i] = tok.lower() == "true"
                return out2

        judged: list[bool | None] | None = None
        last_raw = ""
        message_content: list[dict[str, Any]] = list(blocks)
        n_attempts = 1 + max(0, self.max_retries)
        for attempt in range(n_attempts):
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": message_content}],
            )
            raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            last_raw = raw
            tried = parse_pass_flags(raw, len(criteria))
            if all(v is not None for v in tried):
                judged = tried
                break
            if attempt + 1 < n_attempts:
                message_content = [
                    {
                        "type": "text",
                        "text": (
                            "Your previous output was not valid JSON for this schema.\n"
                            'Return ONLY: {"results":[{"pass":true|false}, ...]} with exactly '
                            f"{len(criteria)} entries."
                        ),
                    },
                    {"type": "text", "text": "Previous output:"},
                    {"type": "text", "text": trunc(raw, 4000)},
                ]
            else:
                judged = tried

        if judged is None:
            judged = [None] * len(criteria)
        judged_bool = [False if v is None else v for v in judged]
        if self.debug_prompts and any(not v for v in judged_bool) and "pass" not in last_raw:
            print("[warn] verifier parse fallback; unresolved marked false", file=sys.stderr)
        score = round(sum(1 for p in judged_bool if p) / len(criteria), 4)
        self.cache[key] = score
        return score


def export_session(
    blob: dict,
    scorer: VerifierScorer,
    *,
    agent_trajectory_style: str = "default",
) -> dict:
    """Build one session: uuid, name, initial_message, learning_units (with rewards)."""

    def parse_action(action: str) -> tuple[str, Any] | None:
        m = _ACTION_RE.match(action)
        if not m:
            return None
        name, payload = m.group(1), m.group(2).strip()
        kind = name.lower()
        if not payload:
            return kind, None
        try:
            if kind in {"write", "edit"}:
                obj = json.loads(payload)
                return (kind, obj) if isinstance(obj, dict) else None
            if kind == "message":
                t = json.loads(payload)
                return (kind, t) if isinstance(t, str) else None
        except json.JSONDecodeError:
            return None
        return kind, payload

    def consolidate_edits_on_trajectory(steps: list[dict]) -> list[dict]:
        """Collapse the inclusive [first edit … last edit] window to merged ``edit`` steps only.

        Every non-edit step inside that window (bash, read, duplicate edits, etc.) is dropped.
        One output ``edit`` per file, with all ``oldText``/``newText`` pairs from that window in order.
        """

        def edit_pairs_from_payload(payload: dict) -> list[dict[str, str]]:
            pairs: list[dict[str, str]] = []
            edits = payload.get("edits")
            if isinstance(edits, list):
                for e in edits:
                    if not isinstance(e, dict):
                        continue
                    ot, nt = e.get("oldText"), e.get("newText")
                    if isinstance(ot, str) and isinstance(nt, str):
                        pairs.append({"oldText": ot, "newText": nt})
            if pairs:
                return pairs
            os_, ns_ = payload.get("old_string"), payload.get("new_string")
            if isinstance(os_, str) and isinstance(ns_, str):
                return [{"oldText": os_, "newText": ns_}]
            return []

        def is_edit_step(step: dict) -> bool:
            act = step.get("action")
            if not isinstance(act, str):
                return False
            pa = parse_action(act)
            return bool(pa and pa[0] == "edit")

        edit_idx = [i for i, st in enumerate(steps) if is_edit_step(st)]
        if not edit_idx:
            return list(steps)

        i_min, i_max = min(edit_idx), max(edit_idx)
        span = steps[i_min : i_max + 1]

        merged_lines: dict[str, list[dict[str, str]]] = {}
        path_order: list[str] = []
        first_step_for_path: dict[str, dict] = {}

        for step in span:
            act = step.get("action")
            if not isinstance(act, str):
                continue
            pa = parse_action(act)
            if not pa or pa[0] != "edit" or not isinstance(pa[1], dict):
                continue
            pl = pa[1]
            rel = pl.get("path") if isinstance(pl.get("path"), str) else pl.get("file_path")
            if not isinstance(rel, str):
                continue
            chunk = edit_pairs_from_payload(pl)
            if not chunk:
                continue
            if rel not in merged_lines:
                path_order.append(rel)
                merged_lines[rel] = []
                first_step_for_path[rel] = step
            merged_lines[rel].extend(chunk)

        if not path_order:
            return list(steps)

        middle: list[dict] = []
        for rel in path_order:
            lines = merged_lines[rel]
            base = dict(first_step_for_path[rel])
            base["action"] = f"edit({json.dumps({'path': rel, 'edits': lines}, ensure_ascii=False)})"
            middle.append(base)

        return list(steps[:i_min]) + middle + list(steps[i_max + 1 :])

    def env_files(step: dict) -> dict[str, str]:
        env = step.get("environment") if isinstance(step, dict) else None
        if not isinstance(env, dict):
            return {}
        ff = env.get("file")
        if isinstance(ff, dict):
            return {k: v for k, v in ff.items() if isinstance(k, str) and isinstance(v, str)}
        if isinstance(ff, list):
            out: dict[str, str] = {}
            for item in ff:
                if isinstance(item, dict):
                    path, content = item.get("path"), item.get("content")
                    if isinstance(path, str) and isinstance(content, str):
                        out[path] = content
            return out
        return {}

    def workflow_criterion_lines(nodes: list[dict]) -> list[str]:
        lines: list[str] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            for v in node.get("verifiers") or []:
                if isinstance(v, dict):
                    c = v.get("criterion")
                    if isinstance(c, str) and c.strip():
                        lines.append(c)
            children = node.get("children")
            if isinstance(children, list):
                lines.extend(workflow_criterion_lines([c for c in children if isinstance(c, dict)]))
        return lines

    def criteria_from_last_workflow(traj: list) -> list[str]:
        for step in reversed(traj):
            env = step.get("environment") if isinstance(step, dict) else None
            wf = env.get("workflow") if isinstance(env, dict) else None
            if not isinstance(wf, list):
                continue
            lines = workflow_criterion_lines([n for n in wf if isinstance(n, dict)])
            if lines:
                return lines
        return []

    def latest_file_blocks(agent_steps: list[dict]) -> list[tuple[str, str]]:
        for step in reversed(agent_steps):
            fm = env_files(step)
            if fm:
                return sorted(fm.items(), key=lambda x: x[0])
        return []

    def inject_plan_tool_results(agent_steps: list[dict], traj: list, wf_upto: int) -> None:
        """Attach workflow JSON to plan / workflow_plan steps (slice traj through wf_upto)."""

        def index_of_plan_action(plan_action: str) -> int | None:
            for i, st in enumerate(traj):
                if isinstance(st, dict) and st.get("actor") == "agent" and st.get("action") == plan_action:
                    return i
            return None

        def workflow_snapshot_through(plan_action: str) -> list[Any] | None:
            start = index_of_plan_action(plan_action)
            if start is None or start > wf_upto:
                return None
            last: list[Any] | None = None
            for st in traj[start : wf_upto + 1]:
                if not isinstance(st, dict):
                    continue
                env = st.get("environment")
                if isinstance(env, dict) and isinstance(env.get("workflow"), list):
                    last = env["workflow"]
            return last

        for step in agent_steps:
            act = step.get("action")
            if not isinstance(act, str):
                continue
            pa = parse_action(act)
            if not pa or pa[0] not in ("plan", "workflow_plan"):
                continue
            wf: Any = None
            if wf_upto >= 0:
                wf = workflow_snapshot_through(act)
            if not isinstance(wf, list):
                env = step.get("environment")
                if isinstance(env, dict):
                    wf = env.get("workflow")
            step["tool_result"] = json.dumps(wf, ensure_ascii=False) if isinstance(wf, list) else "[]"

    def structured_user_messages(unit_index: int, task_units: list, uidx: list[int]) -> list[str]:
        sp = blob.get("system_prompt")
        system_prompt = sp if isinstance(sp, str) else ""
        schemas = blob.get("tool_schemas")
        try:
            tool_json = json.dumps(schemas if schemas is not None else [], ensure_ascii=False)
        except (TypeError, ValueError):
            tool_json = "[]"
        task_t = blob.get("task")
        task_str = task_t if isinstance(task_t, str) else ""
        msgs = [system_prompt, tool_json, task_str]
        if unit_index <= 0:
            return msgs
        for seg in range(unit_index):
            for j in range(uidx[seg] + 1, uidx[seg + 1]):
                u = task_units[j]
                if isinstance(u, dict) and u.get("actor") == "agent":
                    pr = u.get("prompt")
                    if isinstance(pr, str) and pr.strip():
                        msgs.append(pr)
        return msgs

    def rescale_verifier_rewards(units: list[dict]) -> None:
        if len(units) < 2:
            return

        def get_verifier(u: dict) -> float | None:
            r = u.get("reward")
            if not isinstance(r, dict):
                return None
            v = r.get("verifier")
            return float(v) if isinstance(v, (int, float)) else None

        def set_verifier(u: dict, val: float) -> None:
            r = u.get("reward")
            if not isinstance(r, dict):
                r = {}
                u["reward"] = r
            r["verifier"] = round(float(val), 4)

        first = get_verifier(units[0])
        last = get_verifier(units[-1])
        if first is None or last is None:
            return
        set_verifier(units[0], 0.0)
        set_verifier(units[-1], last)
        denom = last - first
        for i in range(1, len(units) - 1):
            cur = get_verifier(units[i])
            if cur is None:
                continue
            scaled = 0.0 if abs(denom) < 1e-12 else ((cur - first) / denom) * last
            set_verifier(units[i], max(0.0, min(1.0, scaled)))

    meta = {"uuid": blob.get("uuid"), "name": blob.get("name")}
    traj = s.trajectory_from_blob(blob)
    if not isinstance(traj, list):
        return _session_dict(meta, None, None, error="bad_trajectory")

    parsed = s.pairs_from_task_units(blob, traj)
    if parsed is not None:
        initial, pairs, user_unit_indices = parsed
        tu = blob.get("task_units")
        task_units = tu if isinstance(tu, list) else None
    else:
        initial, pairs = s.pair_segments(traj)
        user_unit_indices, task_units = None, None

    if not pairs:
        return _session_dict(meta, initial, [])

    crit = criteria_from_last_workflow(traj)
    human_counts = [len(hm) for _, hm in pairs]
    suffix = [0] * len(human_counts)
    run = 0
    for i in range(len(human_counts) - 1, -1, -1):
        run += human_counts[i]
        suffix[i] = run

    prefix = list(initial["steps"]) if initial else []
    agent_history: list[dict] = []
    units: list[dict] = []
    style = agent_trajectory_style
    cumulative = style in ("aggregate", "rewritten")

    for j, (ag, hm) in enumerate(pairs):
        ag, hm = list(ag), list(hm)
        agent_history.extend(ag)
        agent_traj = list(agent_history) if cumulative else list(ag)
        if style == "rewritten":
            agent_traj = consolidate_edits_on_trajectory(agent_traj)
        if hm:
            tail = hm[-1]
        elif ag:
            tail = ag[-1]
        else:
            tail = None
        try:
            wf_upto = traj.index(tail) if tail is not None else (len(traj) - 1 if traj else -1)
        except ValueError:
            wf_upto = len(traj) - 1 if traj else -1
        inject_plan_tool_results(agent_traj, traj, wf_upto)
        if user_unit_indices is not None and task_units is not None:
            umsg = structured_user_messages(j, task_units, user_unit_indices)
        else:
            umsg = [s.user_text(x) for x in prefix]
        files = latest_file_blocks(agent_history)
        units.append(
            {
                "index": j,
                "user_messages": umsg,
                "agent_trajectory": agent_traj,
                "human_trajectory": hm,
                "reward": {"verifier": scorer.score(crit, files), "human": suffix[j]},
            }
        )
        prefix.extend(hm)

    rescale_verifier_rewards(units)
    return _session_dict(meta, initial, units)


def main() -> None:
    p = argparse.ArgumentParser(description="Export trajectories with LLM-rated REINFORCE rewards.")
    p.add_argument("input", nargs="?", default="-", help="Task sessions JSON (file or stdin).")
    p.add_argument("-o", "--output", default="-", help="Output JSON path (default: stdout).")
    p.add_argument("--model", default=DEFAULT_VERIFIER_MODEL, help="Anthropic model for verifier scoring.")
    p.add_argument("--max-tokens", type=int, default=1024, help="Verifier completion max tokens.")
    p.add_argument(
        "--agent_trajectory_style",
        choices=["default", "aggregate", "rewritten"],
        default="aggregate",
        help=(
            "default: this unit's agent steps only; aggregate: cumulative agent steps; "
            "rewritten: cumulative, then replace the inclusive range from first edit to last edit with "
            "one merged edit(...) per file (drops all other steps in that range)."
        ),
    )
    p.add_argument("--no-api-config", action="store_true", help="Skip API config in induce.resolve_anthropic_config.")
    p.add_argument(
        "--no-claude-settings",
        action="store_true",
        help="Skip Claude settings in induce.resolve_anthropic_config.",
    )
    p.add_argument(
        "--no_llm_reward",
        action="store_true",
        help="Skip verifier API; use fixed verifier score 0.5.",
    )
    p.add_argument("--debug-prompts", action="store_true", help="Log verifier request bodies to stderr.")
    args = p.parse_args()

    scorer_kw = dict(model=args.model, max_tokens=int(args.max_tokens), debug_prompts=bool(args.debug_prompts))
    if args.no_llm_reward:
        scorer = VerifierScorer(**scorer_kw, client=None, no_llm_reward=True)
    else:
        cfg = induce.resolve_anthropic_config(
            skip_api_config=bool(args.no_api_config),
            skip_claude_settings=bool(args.no_claude_settings),
        )
        scorer = VerifierScorer(**scorer_kw, client=induce.make_anthropic_client(cfg), no_llm_reward=False)

    raw = json.load(sys.stdin) if args.input == "-" else json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = [export_session(b, scorer, agent_trajectory_style=args.agent_trajectory_style) for b in s.blobs(raw)]
    payload = results[0] if len(results) == 1 else {"sessions": results}
    out = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(out)
    else:
        Path(args.output).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
