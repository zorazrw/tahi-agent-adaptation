#!/usr/bin/env python3
"""LLM-summarize user follow-up actions and attach them as ``summary`` per OPD unit.

Reads the OPD JSON produced by ``export_opd_data.py`` (single-session shape or
``{"sessions": [...]}`` shape) and, for each *group* of learning units, calls
an LLM API to produce a concise golden-answer string that captures the user's
intent across the user's follow-up actions (messages + edits). The same
summary string is then stamped onto every unit in the group.

Group granularity is controlled by ``--granularity``:
    session  one summary per session (default; cheapest)
    task     one summary per consecutive run of units sharing ``task_intent``
    unit     one summary per learning unit (tightest signal, N x cost)

Cached results are written to a sidecar JSON keyed by
``(session_uuid, group_id, content_hash, model)`` so re-runs are free.

Auth/config (OpenAI-compatible API):
    OPENAI_API_KEY        required
    OPENAI_BASE_URL       optional (point at any OpenAI-compatible endpoint)

Usage:
    python scripts/summarize_followups.py opd.json -o opd_summarized.json
    python scripts/summarize_followups.py opd.json --in-place --granularity task
    python scripts/summarize_followups.py opd.json --granularity unit --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm as atqdm

try:
    from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover - openai is a required dep at runtime
    AsyncOpenAI = None  # type: ignore


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CACHE_PATH = "summarize_followups.cache.json"
DEFAULT_CONCURRENCY = 8

SYSTEM_PROMPT = (
    "You produce concise 'golden answers' for self-distillation fine-tuning. "
    "Given the original task instruction and the user's subsequent corrections "
    "(chat messages and file edits), write a single response in the assistant's "
    "voice that, if the agent had produced it up-front, would have satisfied "
    "the user without needing any of those corrections.\n\n"
    "Requirements:\n"
    "- Speak as the assistant performing the task, not as a meta-summarizer.\n"
    "- Preserve concrete constraints: file names, exact colors, numeric values, "
    "axis ranges, tool/library choices, positions, copy text, etc.\n"
    "- If the user asked to remove something, say it should be removed; do not "
    "reintroduce it.\n"
    "- Resolve contradictions in favor of the user's latest instruction.\n"
    "- Keep it under ~400 words. Plain prose; no headings or markdown lists "
    "unless the user explicitly requested them."
)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _cache_key(session_uuid: str, group_id: str, content_hash: str, model: str) -> str:
    return f"{session_uuid}::{group_id}::{model}::{content_hash}"


def _content_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Rendering follow-up actions into LLM-readable prose
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int = 4000) -> str:
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... [truncated {len(text) - limit} chars] ...\n{tail}"


def _render_action(idx: int, action: dict) -> str:
    t = action.get("type", "?")
    rd = action.get("round_index")
    head = f"[{idx}] round={rd} type={t}"

    if t == "follow_up":
        return f"{head}\n  user said: {_truncate(action.get('prompt') or '', 2000)}"

    if t == "file_edit":
        path = action.get("path") or "<unknown>"
        ai = action.get("ai")
        edited = action.get("edited")
        if isinstance(ai, str) and isinstance(edited, str) and ai != edited:
            return (
                f"{head}\n  file: {path}\n  --- agent wrote ---\n"
                f"{_truncate(ai, 1500)}\n  --- human edited to ---\n"
                f"{_truncate(edited, 1500)}"
            )
        if isinstance(edited, str):
            return f"{head}\n  file: {path}\n  human content:\n{_truncate(edited, 1500)}"
        return f"{head}\n  file: {path}"

    if t == "brain_edit":
        mem = action.get("memory")
        sk = action.get("skill")
        body = []
        if mem:
            body.append("memory: " + _truncate(json.dumps(mem, ensure_ascii=False), 1200))
        if sk:
            body.append("skill: " + _truncate(json.dumps(sk, ensure_ascii=False), 1200))
        return head + ("\n  " + "\n  ".join(body) if body else "")

    if t == "edit_workflow":
        wf = action.get("workflow")
        return f"{head}\n  workflow: {_truncate(json.dumps(wf, ensure_ascii=False), 1500)}"

    if t == "edit_verifier":
        crit = action.get("criterion") or (action.get("raw") or {}).get("criterion")
        node = action.get("nodeId")
        if crit:
            return f"{head}\n  node={node} criterion: {_truncate(str(crit), 600)}"
        return f"{head}\n  raw: {_truncate(json.dumps(action.get('raw'), ensure_ascii=False), 800)}"

    return f"{head}\n  payload: {_truncate(json.dumps(action, ensure_ascii=False), 1000)}"


_DEFAULT_TAIL_HEADER = "All remaining user follow-up actions from here on out"
_DEFAULT_JOB_INSTRUCTION = (
    "Write the single assistant response that, given the conversation up to "
    "the latest follow-up above, would have produced an outcome that needs "
    "none of the remaining corrections. Speak as the assistant."
)


def _build_user_prompt(
    *,
    initial_task_instruction: str,
    task_intent: str | None,
    user_messages: list[str],
    followup_actions: list[dict],
    tail_header: str = _DEFAULT_TAIL_HEADER,
    job_instruction: str = _DEFAULT_JOB_INSTRUCTION,
) -> str:
    parts: list[str] = []
    parts.append("# Original task")
    parts.append(_truncate(initial_task_instruction or "", 3000))
    if task_intent:
        parts.append("\n# Current sub-task")
        parts.append(_truncate(task_intent, 800))
    if user_messages:
        parts.append("\n# Latest user follow-up (the prompt that triggered this training unit)")
        for um in user_messages:
            parts.append("- " + _truncate(um, 2000))
    parts.append(f"\n# {tail_header}")
    if not followup_actions:
        parts.append("(none)")
    else:
        for i, a in enumerate(followup_actions):
            parts.append(_render_action(i, a))
    parts.append("\n# Your job\n" + job_instruction)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _make_client() -> Any:
    if AsyncOpenAI is None:
        raise RuntimeError(
            "openai package is not installed. `pip install openai` (see scripts/requirements.txt)."
        )
    kwargs: dict[str, Any] = {}
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


async def _summarize_once_async(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
) -> str:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:  # pragma: no cover - network/error path
            last_err = e
            sleep_s = 2 ** attempt
            print(
                f"  LLM call failed (attempt {attempt + 1}/{max_retries}): {e}; sleeping {sleep_s}s",
                file=sys.stderr,
            )
            await asyncio.sleep(sleep_s)
    raise RuntimeError(f"LLM summarization failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _sessions_view(raw: Any) -> list[dict]:
    if isinstance(raw, dict) and isinstance(raw.get("sessions"), list):
        return [s for s in raw["sessions"] if isinstance(s, dict)]
    if isinstance(raw, dict) and isinstance(raw.get("learning_units"), list):
        return [raw]
    return []


def _dedupe_actions(actions_list: list[list[dict]]) -> list[dict]:
    """Concatenate per-unit ``followup_actions`` lists, deduping by content hash."""
    out: list[dict] = []
    seen: set[str] = set()
    for sub in actions_list:
        for a in sub or []:
            if not isinstance(a, dict):
                continue
            sig = _content_hash(a)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(a)
    return out


def _group_units(units: list[dict], granularity: str) -> list[tuple[str, list[dict]]]:
    """Bucket units into ``(group_id, [units])`` according to granularity.

    - ``unit``    : one group per unit (legacy behavior).
    - ``task``    : group consecutive units that share ``task_intent``.
    - ``session`` : a single group containing every unit.
    """
    valid = [u for u in units if isinstance(u, dict)]
    if granularity == "unit":
        return [(f"unit:{int(u.get('index', i))}", [u]) for i, u in enumerate(valid)]
    if granularity == "session":
        return [("session", valid)] if valid else []
    if granularity == "task":
        groups: list[tuple[str, list[dict]]] = []
        current_intent: object = object()  # sentinel
        current_bucket: list[dict] = []
        current_id = ""
        for u in valid:
            intent = u.get("task_intent")
            if intent != current_intent:
                if current_bucket:
                    groups.append((current_id, current_bucket))
                current_intent = intent
                tag = (intent or "untitled").strip().split("\n", 1)[0][:48].replace(" ", "_")
                current_id = f"task:{tag}:{len(groups)}"
                current_bucket = []
            current_bucket.append(u)
        if current_bucket:
            groups.append((current_id, current_bucket))
        return groups
    raise ValueError(f"Unknown granularity: {granularity!r}")


def _build_group_prompt(
    *,
    initial_task: str,
    units: list[dict],
    granularity: str,
) -> tuple[str, str]:
    """Return ``(prompt, scope_label)`` for a unit group.

    For session/task scope the per-unit ``user_messages`` and per-unit
    ``followup_actions`` are aggregated and deduped, and the prompt headings
    are reworded so the LLM doesn't expect a single "latest follow-up" anchor.
    """
    if granularity == "unit":
        u = units[0]
        return (
            _build_user_prompt(
                initial_task_instruction=initial_task,
                task_intent=u.get("task_intent"),
                user_messages=u.get("user_messages") or [],
                followup_actions=u.get("followup_actions") or [],
            ),
            f"unit {u.get('index')}",
        )

    # Aggregate across units.
    all_followups = _dedupe_actions([u.get("followup_actions") or [] for u in units])
    # Pull out task_intent (only meaningful for granularity=task).
    task_intent = units[0].get("task_intent") if granularity == "task" else None

    if granularity == "task":
        scope_label = f"task '{(task_intent or 'untitled')[:40]}' ({len(units)} units)"
        tail_header = "All user follow-up actions during this sub-task"
        job_instruction = (
            "Write the single assistant response that, if produced up-front in "
            "response to the original task and the current sub-task, would have "
            "satisfied the user without needing any of the corrections below. "
            "Speak as the assistant."
        )
    else:  # session
        scope_label = f"whole session ({len(units)} units)"
        tail_header = "All user follow-up actions across this session"
        job_instruction = (
            "Write the single assistant response that, if produced up-front in "
            "response to the original task, would have satisfied the user without "
            "needing any of the corrections below. Speak as the assistant."
        )

    prompt = _build_user_prompt(
        initial_task_instruction=initial_task,
        task_intent=task_intent,
        user_messages=[],  # no single latest-follow-up anchor at this scope
        followup_actions=all_followups,
        tail_header=tail_header,
        job_instruction=job_instruction,
    )
    return prompt, scope_label


async def _annotate_session_async(
    session: dict,
    *,
    client: Any,
    model: str,
    system_prompt: str,
    cache: dict[str, str],
    cache_path: Path,
    cache_lock: asyncio.Lock,
    dry_run: bool,
    overwrite: bool,
    semaphore: asyncio.Semaphore,
    granularity: str,
) -> tuple[int, int, int]:
    """Mutates ``session`` to add ``summary`` to each learning unit.

    Units are bucketed into groups according to ``granularity`` (see
    :func:`_group_units`); one LLM request is made per group, and the
    resulting summary is stamped onto every unit in that group. Groups
    are dispatched concurrently under ``semaphore``; the cache is
    persisted after each response so a Ctrl+C only loses in-flight work.

    Returns ``(n_written, n_cached, n_skipped)`` counted in *units*.
    """
    n_written = n_cached = n_skipped = 0
    session_uuid = str(session.get("uuid") or "")
    initial_task = session.get("initial_task_instruction") or ""
    units = session.get("learning_units") or []

    groups = _group_units(units, granularity)

    pending: list[tuple[str, list[dict], str, str]] = []  # (group_id, units, prompt, key)

    for group_id, gunits in groups:
        # If every unit in the group already has a summary and we're not
        # overwriting, treat the whole group as cache-skip.
        if not overwrite and all(u.get("summary") for u in gunits):
            n_skipped += len(gunits)
            continue

        # Drop groups whose units have nothing to summarize at all.
        if not any(
            (u.get("followup_actions") or []) or (u.get("user_messages") or [])
            for u in gunits
        ):
            n_skipped += len(gunits)
            continue

        prompt, scope_label = _build_group_prompt(
            initial_task=initial_task,
            units=gunits,
            granularity=granularity,
        )
        key = _cache_key(
            session_uuid,
            group_id,
            _content_hash({"system": system_prompt, "user": prompt}),
            model,
        )

        if key in cache and not overwrite:
            summary = cache[key]
            for u in gunits:
                if u.get("summary") and not overwrite:
                    continue
                u["summary"] = summary
                n_cached += 1
            continue

        if dry_run:
            print(
                f"[dry-run] would summarize {session_uuid} {scope_label} "
                f"({len(gunits)} units, prompt~{len(prompt)} chars)",
                file=sys.stderr,
            )
            n_skipped += len(gunits)
            continue

        pending.append((group_id, gunits, prompt, key))

    if not pending:
        return n_written, n_cached, n_skipped

    async def _worker(
        group_id: str, gunits: list[dict], prompt: str, key: str
    ) -> tuple[int, Exception | None]:
        try:
            async with semaphore:
                summary = await _summarize_once_async(
                    client,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                )
        except Exception as e:
            return 0, e
        applied = 0
        for u in gunits:
            if u.get("summary") and not overwrite:
                continue
            u["summary"] = summary
            applied += 1
        async with cache_lock:
            cache[key] = summary
            _save_cache(cache_path, cache)
        return applied, None

    tasks = [
        asyncio.create_task(
            _worker(gid, gunits, p, k),
            name=f"sum:{session_uuid[:8]}:{gid}",
        )
        for gid, gunits, p, k in pending
    ]

    results = await atqdm.gather(
        *tasks,
        desc=f"summarize {session_uuid[:8]} ({len(tasks)} groups)",
    )
    for applied, err in results:
        if err is not None:
            print(f"  group failed: {err}", file=sys.stderr)
        else:
            n_written += applied

    return n_written, n_cached, n_skipped


async def _run(args: argparse.Namespace) -> int:
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    sessions = _sessions_view(raw)
    if not sessions:
        print("No sessions / learning_units found in input.", file=sys.stderr)
        return 1

    cache_path = Path(args.cache)
    cache = _load_cache(cache_path)
    system_prompt = args.system_prompt or SYSTEM_PROMPT
    client = None if args.dry_run else _make_client()

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    cache_lock = asyncio.Lock()

    totals = [0, 0, 0]
    try:
        for s in sessions:
            nw, nc, ns = await _annotate_session_async(
                s,
                client=client,
                model=args.model,
                system_prompt=system_prompt,
                cache=cache,
                cache_path=cache_path,
                cache_lock=cache_lock,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                semaphore=semaphore,
                granularity=args.granularity,
            )
            totals[0] += nw
            totals[1] += nc
            totals[2] += ns
            print(
                f"session {s.get('uuid')}: wrote {nw}, cached {nc}, skipped {ns}",
                file=sys.stderr,
            )
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass

    text = json.dumps(raw, indent=2, ensure_ascii=False) + "\n"
    if args.in_place:
        Path(args.input).write_text(text, encoding="utf-8")
    elif args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    print(
        f"Done. total wrote={totals[0]} cached={totals[1]} skipped={totals[2]}",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Summarize user follow-up actions and attach as `summary` per OPD learning unit."
    )
    p.add_argument("input", help="OPD JSON produced by export_opd_data.py")
    out_group = p.add_mutually_exclusive_group()
    out_group.add_argument("-o", "--output", default=None, help="Output path (default: stdout)")
    out_group.add_argument("--in-place", action="store_true", help="Overwrite the input file")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI-compatible model name (default: {DEFAULT_MODEL})")
    p.add_argument("--system-prompt", default=None, help="Override the default summarizer system prompt")
    p.add_argument(
        "--cache",
        default=DEFAULT_CACHE_PATH,
        help=f"Path to summary cache JSON (default: {DEFAULT_CACHE_PATH})",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent in-flight LLM requests (default: {DEFAULT_CONCURRENCY})",
    )
    p.add_argument(
        "--granularity",
        choices=("session", "task", "unit"),
        default="session",
        help=(
            "Summarization scope: 'session' (one summary per session, applied to "
            "every unit -- cheap and usually sufficient), 'task' (one summary per "
            "consecutive run of units sharing task_intent), or 'unit' (one summary "
            "per learning unit, tightest signal but N times more LLM calls). "
            "Default: session."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would be summarized, do not call the API")
    p.add_argument("--overwrite", action="store_true", help="Recompute summaries even if cached or present")
    args = p.parse_args()

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
