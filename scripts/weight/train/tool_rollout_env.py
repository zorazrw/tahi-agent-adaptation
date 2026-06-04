"""Tool-using rollout environment (reusable by OPD and REINFORCE).

This module defines a standalone, *objective-agnostic* tool-using rollout
environment. A student model runs a real multi-turn agent loop, executing file
tools (``read``/``write``/``edit``/``grep``/``find``/``ls``/``bash``) inside a
local ephemeral sandbox. The same environment serves both collection paths:

* OPD (distillation): grading is irrelevant, so ``reward_fn`` defaults to a
  zero-reward stub. The value of the env is the on-policy multi-turn trajectory.
* REINFORCE (reward-based): supply a ``sandbox_reward_fn`` that grades the
  final filesystem state (or a rubric/judge). See :class:`SandboxAgentToolEnv`.

The design follows the Tinker cookbook idioms (see ``tinker_cookbook.tool_use``
and ``tinker_cookbook.rl``):

* file tools are stateful ``@tool`` methods bound to a per-episode sandbox,
* the env is built on :class:`~tinker_cookbook.tool_use.AgentToolMessageEnv`
  wrapped by :class:`~tinker_cookbook.rl.message_env.EnvFromMessageEnv`,
* a *pickleable* :class:`~tinker_cookbook.rl.types.EnvGroupBuilder` creates the
  sandboxes lazily in ``make_envs()`` and tears them down in ``cleanup()``.

Wiring the trajectories into OPD top-K distillation and REINFORCE advantage /
datum construction is intentionally **out of scope** here (deferred), but the
env is shaped to make that step straightforward.
"""

from __future__ import annotations

import asyncio
import fnmatch
import glob as _glob
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Optional

from pydantic import BaseModel

from tinker_cookbook import model_info
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use import (
    AgentToolMessageEnv,
    ToolResult,
    error_tool_result,
    simple_tool_result,
    tool,
)

logger = logging.getLogger(__name__)

# Output cap applied to every tool result to keep observations bounded.
_MAX_TOOL_OUTPUT_CHARS = 50_000

RewardResult = tuple[float, dict[str, float]]
MessageRewardFn = Callable[[list[Message]], Awaitable[RewardResult]]
SandboxRewardFn = Callable[[list[Message], "WorkspaceSandbox"], Awaitable[RewardResult]]


async def zero_reward(_history: list[Message]) -> RewardResult:
    """Default reward for the OPD/distillation path: no reward signal."""
    return 0.0, {}


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class WorkspaceSandbox:
    """A per-episode ephemeral working directory.

    Creates a ``tempfile.mkdtemp()`` root, optionally seeded with
    ``{path: content}`` files, and confines every file/command operation to
    that root. Single-use: call :meth:`cleanup` (idempotent) when done.
    """

    def __init__(
        self,
        seed_files: dict[str, str] | None = None,
        *,
        prefix: str = "tool_rollout_",
        bash_timeout_s: int = 20,
    ) -> None:
        # realpath so relative-path computations (grep/find/ls) stay clean on
        # platforms where the temp dir lives behind a symlink (e.g. macOS /var).
        self.root = os.path.realpath(tempfile.mkdtemp(prefix=prefix))
        self.bash_timeout_s = bash_timeout_s
        self._closed = False
        for rel, content in (seed_files or {}).items():
            self.write_text(rel, content)

    def _resolve(self, path: str | None) -> str:
        """Resolve ``path`` to an absolute path confined within the sandbox.

        Absolute-looking or backslash paths are normalized into the sandbox so
        that seeding and access use a consistent key space. Raises ``ValueError``
        if the resolved path escapes the sandbox root.
        """
        rel = (path or ".").strip().replace("\\", "/").lstrip("/")
        candidate = os.path.normpath(os.path.join(self.root, rel))
        real_root = os.path.realpath(self.root)
        real = os.path.realpath(candidate)
        if real != real_root and not real.startswith(real_root + os.sep):
            raise ValueError(f"Path escapes sandbox: {path!r}")
        return real

    def write_text(self, path: str, content: str) -> None:
        real = self._resolve(path)
        os.makedirs(os.path.dirname(real) or self.root, exist_ok=True)
        with open(real, "w", encoding="utf-8") as f:
            f.write(content)

    def read_text(self, path: str) -> str:
        real = self._resolve(path)
        with open(real, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def run_command(self, command: str, timeout: int | None = None) -> tuple[int, str, str]:
        """Run a shell command confined to the sandbox root (blocking, headless).

        The subprocess is forced into a headless GUI environment so agent code
        that calls e.g. ``matplotlib.pyplot.show()`` does not pop a real window
        and block the rollout until a human closes it. ``MPLBACKEND=Agg`` makes
        matplotlib non-interactive (``show()`` is a no-op; ``savefig()`` still
        works); ``QT_QPA_PLATFORM=offscreen`` and clearing ``DISPLAY`` cover
        Qt/X-based GUIs.
        """
        env = {**os.environ, "MPLBACKEND": "Agg", "QT_QPA_PLATFORM": "offscreen"}
        env.pop("DISPLAY", None)
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout if timeout is not None else self.bash_timeout_s,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def snapshot(self) -> dict[str, str]:
        """Return ``{relpath: content}`` for all text files (for future grading)."""
        out: dict[str, str] = {}
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, self.root)
                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        out[rel] = f.read()
                except OSError:
                    continue
        return out

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self.root, ignore_errors=True)


def _truncate(text: str, limit: int = _MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"


# ---------------------------------------------------------------------------
# File tools (stateful, bound to one sandbox)
# ---------------------------------------------------------------------------


class EditOp(BaseModel):
    """One targeted text replacement for the ``edit`` tool."""

    oldText: str
    newText: str


class FileToolset:
    """File tools bound to a single :class:`WorkspaceSandbox`.

    Tool names and parameter shapes mirror the Pi agent tools (see
    ``PI_TOOL_SCHEMAS`` in ``scripts/tasks/export_task_sessions.py`` and
    ``src/electron/libs/runner.ts``) so the rendered tool schemas match the
    chat template the model was trained under. Pass :meth:`tools` to the env.
    """

    def __init__(
        self,
        sandbox: WorkspaceSandbox,
        *,
        enable_bash: bool = True,
        bash_timeout_s: int = 20,
    ) -> None:
        self.sandbox = sandbox
        self._enable_bash = enable_bash
        self._bash_timeout_s = bash_timeout_s

    def tools(self) -> list:
        """Return the bound tool list (``bash`` included only when enabled).

        ``workflow_plan`` and ``ask_user_question`` are always included so the
        executable tool surface matches the Pi agent env's advertised schemas
        (otherwise the model would hit a ``tool_not_found`` error mid-rollout).
        """
        bound = [self.read, self.write, self.edit, self.grep, self.find, self.ls]
        if self._enable_bash:
            bound.append(self.bash)
        bound.extend([self.workflow_plan, self.ask_user_question])
        return bound

    def _iter_files(self, base: str) -> list[str]:
        if os.path.isfile(base):
            return [base]
        results: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            if ".git" in dirnames:
                dirnames.remove(".git")
            for fn in filenames:
                results.append(os.path.join(dirpath, fn))
        return results

    @tool
    async def read(
        self,
        path: Annotated[str, "Path to the file to read (relative or absolute)"],
        offset: Annotated[int, "Line number to start reading from (1-indexed)"] = 1,
        limit: Annotated[int, "Maximum number of lines to read"] = 2000,
    ) -> ToolResult:
        """Read the contents of a file. Use offset/limit for large files."""
        try:
            text = self.sandbox.read_text(path)
        except FileNotFoundError:
            return error_tool_result(f"File not found: {path}", name="read", error_type="not_found")
        except OSError as e:
            return error_tool_result(str(e), name="read", error_type="read_failed")
        lines = text.splitlines()
        start = max(1, offset) - 1
        selected = lines[start : start + max(1, limit)]
        return simple_tool_result(_truncate("\n".join(selected)), name="read")

    @tool
    async def write(
        self,
        path: Annotated[str, "Path to the file to write (relative or absolute)"],
        content: Annotated[str, "Content to write to the file"],
    ) -> ToolResult:
        """Write content to a file, creating parent directories as needed."""
        try:
            self.sandbox.write_text(path, content)
        except (OSError, ValueError) as e:
            return error_tool_result(str(e), name="write", error_type="write_failed")
        n_lines = content.count("\n") + 1 if content else 0
        return simple_tool_result(
            f"Wrote {len(content)} bytes ({n_lines} lines) to {path}", name="write"
        )

    @tool
    async def edit(
        self,
        path: Annotated[str, "Path to the file to edit (relative or absolute)"],
        edits: Annotated[list[EditOp], "Targeted, non-overlapping oldText->newText replacements"],
    ) -> ToolResult:
        """Edit a file via exact, unique text replacements."""
        try:
            text = self.sandbox.read_text(path)
        except FileNotFoundError:
            return error_tool_result(f"File not found: {path}", name="edit", error_type="not_found")
        except OSError as e:
            return error_tool_result(str(e), name="edit", error_type="read_failed")

        new_text = text
        applied = 0
        for raw in edits:
            # @tool validates and passes dicts (model_dump); tolerate models too.
            old = raw["oldText"] if isinstance(raw, dict) else raw.oldText
            rep = raw["newText"] if isinstance(raw, dict) else raw.newText
            count = new_text.count(old)
            if count == 0:
                return error_tool_result(
                    f"oldText not found: {old[:80]!r}", name="edit", error_type="no_match"
                )
            if count > 1:
                return error_tool_result(
                    f"oldText not unique ({count} matches): {old[:80]!r}",
                    name="edit",
                    error_type="ambiguous",
                )
            new_text = new_text.replace(old, rep, 1)
            applied += 1
        try:
            self.sandbox.write_text(path, new_text)
        except (OSError, ValueError) as e:
            return error_tool_result(str(e), name="edit", error_type="write_failed")
        return simple_tool_result(f"Applied {applied} edit(s) to {path}", name="edit")

    @tool
    async def grep(
        self,
        pattern: Annotated[str, "Search pattern (regex or literal string)"],
        path: Annotated[str, "Directory or file to search (default: current directory)"] = ".",
        glob: Annotated[Optional[str], "Filter files by glob pattern, e.g. '*.ts'"] = None,
        ignoreCase: Annotated[bool, "Case-insensitive search"] = False,
        literal: Annotated[bool, "Treat pattern as a literal string instead of regex"] = False,
        limit: Annotated[int, "Maximum number of matches to return"] = 100,
    ) -> ToolResult:
        """Search file contents for a pattern. Returns 'relpath:lineno:line'."""
        flags = re.IGNORECASE if ignoreCase else 0
        raw_pat = re.escape(pattern) if literal else pattern
        try:
            rx = re.compile(raw_pat, flags)
        except re.error as e:
            return error_tool_result(f"Invalid regex: {e}", name="grep", error_type="bad_pattern")
        try:
            base = self.sandbox._resolve(path)
        except ValueError as e:
            return error_tool_result(str(e), name="grep", error_type="bad_path")

        matches: list[str] = []
        for full in self._iter_files(base):
            if glob and not (
                fnmatch.fnmatch(os.path.basename(full), glob)
                or fnmatch.fnmatch(os.path.relpath(full, base), glob)
            ):
                continue
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    contents = f.read()
            except OSError:
                continue
            rel = os.path.relpath(full, self.sandbox.root)
            for i, line in enumerate(contents.splitlines(), 1):
                if rx.search(line):
                    matches.append(f"{rel}:{i}:{line[:500]}")
                    if len(matches) >= limit:
                        break
            if len(matches) >= limit:
                break
        body = "\n".join(matches) if matches else "No matches found."
        return simple_tool_result(_truncate(body), name="grep")

    @tool
    async def find(
        self,
        pattern: Annotated[str, "Glob pattern, e.g. '*.ts', '**/*.json'"],
        path: Annotated[str, "Directory to search in (default: current directory)"] = ".",
        limit: Annotated[int, "Maximum number of results"] = 1000,
    ) -> ToolResult:
        """Find files by glob pattern. Returns paths relative to the search dir."""
        try:
            base = self.sandbox._resolve(path)
        except ValueError as e:
            return error_tool_result(str(e), name="find", error_type="bad_path")
        if not os.path.isdir(base):
            return error_tool_result(f"Not a directory: {path}", name="find", error_type="not_dir")
        results = [
            p
            for p in _glob.glob(pattern, root_dir=base, recursive=True)
            if os.path.isfile(os.path.join(base, p))
        ]
        results = sorted(results)[: max(1, limit)]
        body = "\n".join(results) if results else "No files found."
        return simple_tool_result(_truncate(body), name="find")

    @tool
    async def ls(
        self,
        path: Annotated[str, "Directory to list (default: current directory)"] = ".",
        limit: Annotated[int, "Maximum number of entries to return"] = 500,
    ) -> ToolResult:
        """List directory contents, alphabetically, with '/' for directories."""
        try:
            base = self.sandbox._resolve(path)
        except ValueError as e:
            return error_tool_result(str(e), name="ls", error_type="bad_path")
        if not os.path.isdir(base):
            return error_tool_result(f"Not a directory: {path}", name="ls", error_type="not_dir")
        entries = sorted(os.listdir(base))[: max(1, limit)]
        rendered = [
            name + ("/" if os.path.isdir(os.path.join(base, name)) else "") for name in entries
        ]
        return simple_tool_result("\n".join(rendered) if rendered else "(empty)", name="ls")

    @tool
    async def bash(
        self,
        command: Annotated[str, "Bash command to execute"],
        timeout: Annotated[Optional[int], "Timeout in seconds"] = None,
    ) -> ToolResult:
        """Execute a bash command in the sandbox working directory."""
        if not self._enable_bash:
            return error_tool_result("bash is disabled", name="bash", error_type="disabled")
        t = timeout if timeout is not None else self._bash_timeout_s
        try:
            rc, out, err = await asyncio.to_thread(self.sandbox.run_command, command, t)
        except subprocess.TimeoutExpired:
            return error_tool_result(
                f"Command timed out after {t}s", name="bash", error_type="timeout"
            )
        except OSError as e:
            return error_tool_result(str(e), name="bash", error_type="exec_failed")
        body = f"exit_code: {rc}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"
        return simple_tool_result(_truncate(body), name="bash")

    @tool
    async def workflow_plan(
        self,
        tasks: Annotated[list, "Top-level workflow steps (description/outputFiles/verifiers/children)"],
    ) -> ToolResult:
        """Register a hierarchical workflow plan of 3-5 main steps."""
        # No-op registration that mirrors the Pi planning phase verbatim: after
        # planning the agent must STOP, and the orchestrator (here the agentic
        # OPD rollout driver) then issues one "Proceed with: ..." user turn per
        # planned step. The registered plan is recovered from this tool call's
        # arguments by ``_extract_workflow_plan_tasks`` in run_opd.py, so the
        # step queries are derived from the model's *own* on-policy plan.
        return simple_tool_result(
            "Workflow plan registered. Stop now. Do not execute any steps.",
            name="workflow_plan",
        )

    @tool
    async def ask_user_question(
        self,
        questions: Annotated[list, "Questions to ask the user"],
        answers: Annotated[Optional[list], "Pre-filled answers (unused; autonomous rollout)"] = None,
    ) -> ToolResult:
        """Ask the user a question. In autonomous rollout the user always proceeds."""
        return simple_tool_result("proceed", name="ask_user_question")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class SandboxAgentToolEnv(AgentToolMessageEnv):
    """``AgentToolMessageEnv`` that can grade against the episode's sandbox.

    Stock ``AgentToolMessageEnv.reward_fn`` only receives the message history
    (the cookbook flags this with a ``TODO`` about stateful grading). For
    REINFORCE we usually need the *final filesystem state*, which is not fully
    represented in the transcript. When ``sandbox_reward_fn`` is provided, we
    wrap it into the message-only ``reward_fn`` contract via a closure over the
    owning sandbox, so the parent's ``reward_fn(history)`` call can read files.
    OPD just uses the default zero reward and ignores the sandbox.
    """

    sandbox: Optional[WorkspaceSandbox] = None
    sandbox_reward_fn: Optional[SandboxRewardFn] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sandbox_reward_fn is not None:
            sandbox_reward_fn = self.sandbox_reward_fn
            sandbox = self.sandbox

            async def _wrapped(history: list[Message]) -> RewardResult:
                return await sandbox_reward_fn(history, sandbox)

            self.reward_fn = _wrapped


def build_tool_rollout_env(
    *,
    renderer: Renderer,
    tools: list,
    initial_messages: list[Message],
    sandbox: WorkspaceSandbox,
    max_turns: int = 12,
    reward_fn: MessageRewardFn | None = None,
    sandbox_reward_fn: SandboxRewardFn | None = None,
    failed_parse_reward: float = -0.1,
    max_trajectory_tokens: int | None = None,
    max_generation_tokens: int | None = None,
    context_overflow_reward: float = -0.1,
) -> EnvFromMessageEnv:
    """Build a token-level ``Env`` for a tool-using agent over one sandbox.

    Mirrors ``tinker_cookbook.tool_use.build_agent_tool_env`` but uses
    :class:`SandboxAgentToolEnv` so REINFORCE can grade filesystem state.
    """
    msg_env = SandboxAgentToolEnv(
        tools=tools,
        initial_messages=initial_messages,
        max_turns=max_turns,
        reward_fn=reward_fn or zero_reward,
        sandbox=sandbox,
        sandbox_reward_fn=sandbox_reward_fn,
    )
    return EnvFromMessageEnv(
        renderer=renderer,
        message_env=msg_env,
        failed_parse_reward=failed_parse_reward,
        max_trajectory_tokens=max_trajectory_tokens,
        max_generation_tokens=max_generation_tokens,
        context_overflow_reward=context_overflow_reward,
    )


# ---------------------------------------------------------------------------
# Group builder + dataset (the seam OPD / REINFORCE plug into later)
# ---------------------------------------------------------------------------


class ToolRolloutEnvGroupBuilder(EnvGroupBuilder):
    """Pickleable builder for a group of tool-using episodes.

    Stores only picklable configuration. Heavy, per-episode resources
    (sandboxes, tools, renderer) are constructed lazily in :meth:`make_envs`
    (which runs in the rollout worker) and released in :meth:`cleanup`. This
    keeps the builder safe to ship to ``ProcessPoolExecutor`` / Ray workers.

    ``prompt_messages`` is the conversation *without* the tool prefix (history +
    user turns, in renderer ``Message`` form). The tool prefix is injected here
    from the bound toolset so the advertised schemas always match the executor.
    """

    def __init__(
        self,
        *,
        prompt_messages: list[Message],
        model_name: str,
        renderer_name: str | None = None,
        seed_files: dict[str, str] | None = None,
        system_prompt: str = "",
        group_size: int = 1,
        max_turns: int = 12,
        enable_bash: bool = True,
        bash_timeout_s: int = 20,
        max_trajectory_tokens: int | None = None,
        max_generation_tokens: int | None = None,
        reward_fn: MessageRewardFn | None = None,
        sandbox_reward_fn: SandboxRewardFn | None = None,
        tag: str | None = None,
    ) -> None:
        self.prompt_messages = prompt_messages
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.seed_files = seed_files or {}
        self.system_prompt = system_prompt
        self.group_size = group_size
        self.max_turns = max_turns
        self.enable_bash = enable_bash
        self.bash_timeout_s = bash_timeout_s
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.reward_fn = reward_fn
        self.sandbox_reward_fn = sandbox_reward_fn
        self.tag = tag
        # Populated in make_envs(); empty at pickle time so the builder stays
        # picklable for distributed rollout executors.
        self._sandboxes: list[WorkspaceSandbox] = []

    def _build_renderer(self) -> Renderer:
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        return get_renderer(renderer_name, get_tokenizer(self.model_name))

    async def make_envs(self) -> Sequence[Env]:
        renderer = self._build_renderer()
        envs: list[Env] = []
        for _ in range(self.group_size):
            sandbox = WorkspaceSandbox(self.seed_files, bash_timeout_s=self.bash_timeout_s)
            toolset = FileToolset(
                sandbox, enable_bash=self.enable_bash, bash_timeout_s=self.bash_timeout_s
            )
            tools = toolset.tools()
            initial_messages = renderer.create_conversation_prefix_with_tools(
                [t.to_spec() for t in tools], self.system_prompt
            ) + list(self.prompt_messages)
            env = build_tool_rollout_env(
                renderer=renderer,
                tools=tools,
                initial_messages=initial_messages,
                sandbox=sandbox,
                max_turns=self.max_turns,
                reward_fn=self.reward_fn,
                sandbox_reward_fn=self.sandbox_reward_fn,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
            )
            self._sandboxes.append(sandbox)
            envs.append(env)
        return envs

    async def cleanup(self) -> None:
        for sandbox in self._sandboxes:
            try:
                sandbox.cleanup()
            except Exception:  # noqa: BLE001 - cleanup must never raise
                logger.warning("Failed to clean up sandbox at %s", sandbox.root, exc_info=True)
        self._sandboxes = []

    def logging_tags(self) -> list[str]:
        return [self.tag] if self.tag else []


class ToolRolloutDataset(RLDataset):
    """Thin ``RLDataset`` over a list of :class:`ToolRolloutEnvGroupBuilder`."""

    def __init__(self, builders: Sequence[ToolRolloutEnvGroupBuilder], batch_size: int) -> None:
        self.builders = list(builders)
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        return self.builders[start : start + self.batch_size]

    def __len__(self) -> int:
        return max(1, len(self.builders) // self.batch_size)


__all__ = [
    "WorkspaceSandbox",
    "FileToolset",
    "EditOp",
    "SandboxAgentToolEnv",
    "build_tool_rollout_env",
    "ToolRolloutEnvGroupBuilder",
    "ToolRolloutDataset",
    "zero_reward",
    "RewardResult",
    "MessageRewardFn",
    "SandboxRewardFn",
]
