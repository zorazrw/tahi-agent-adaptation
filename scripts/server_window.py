"""Tinker proxy with rolling training window (latest K submitted sessions).

Same HTTP API as ``server.py``, but keeps ordered session history and trains on
the latest ``training_window_sessions`` each round instead of only the batch
that triggered training. Training rounds are skipped until the window is full
(e.g. with ``training_window_sessions: 4``, the first three submissions are
recorded but do not trigger training).

Run::

    python server_window.py --config config_window.yaml

Use a separate ``state_path`` in that config so checkpoints do not overwrite
the default ``server.py`` experiment state.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import uvicorn
import yaml

from server import Config, Server

log = logging.getLogger(__name__)

_TRAIN_TRIGGER = object()


@dataclass
class WindowConfig(Config):
    """``Config`` plus rolling session window for training rounds."""

    training_window_sessions: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WindowConfig":
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        valid_keys = {fld.name for fld in fields(cls)}
        unknown = set(raw) - valid_keys
        if unknown:
            log.warning("Ignoring unknown config keys in %s: %s", path, ", ".join(sorted(unknown)))
        return cls(**{k: v for k, v in raw.items() if k in valid_keys})


class WindowServer(Server):
    """``Server`` that trains on the latest K sessions from full submission history."""

    def __init__(self, config: WindowConfig):
        if config.training_window_sessions < 1:
            raise ValueError("training_window_sessions must be >= 1")
        super().__init__(config)
        # (order, session_id, payload) in arrival order
        self._session_history: list[tuple[int, str | None, dict[str, Any]]] = []
        self._next_order = 0

    async def _process_sessions(self) -> None:
        while True:
            session_data = await self.sessions_queue.get()
            try:
                if self.config.mode not in ("dpo", "opd", "reinforce", "grpo_opd"):
                    raise ValueError(
                        f"Unknown training mode: {self.config.mode} "
                        "(expected one of 'dpo', 'opd', 'reinforce', 'grpo_opd')"
                    )

                order = self._next_order
                self._next_order += 1
                session_id = session_data.get("uuid")
                self._session_history.append((order, session_id, session_data))

                unit_count = len(session_data.get("task_units", []))
                if unit_count == 0:
                    unit_count = len(session_data.get("learning_units", []))

                k = self.config.training_window_sessions
                history_len = len(self._session_history)
                if history_len < k:
                    log.info(
                        "Session processed: order=%d id=%s mode=%s units=%d "
                        "history=%d window=%d — window not full, skipping training",
                        order,
                        session_id,
                        self.config.mode,
                        unit_count,
                        history_len,
                        k,
                    )
                    continue

                await self.training_queue.put(_TRAIN_TRIGGER)
                pending = self.training_queue.qsize()
                log.info(
                    "Session processed: order=%d id=%s mode=%s units=%d "
                    "history=%d window=%d pending_triggers=%d/%d",
                    order,
                    session_id,
                    self.config.mode,
                    unit_count,
                    history_len,
                    k,
                    pending,
                    self.config.update_every_n_sessions,
                )
                if pending >= self.config.update_every_n_sessions:
                    self.training_event.set()
            except Exception:
                log.exception(
                    "Session processing failed: id=%s mode=%s",
                    session_data.get("uuid"),
                    self.config.mode,
                )
            finally:
                self.sessions_queue.task_done()

    def _drain_queue(self) -> list:
        """Consume training triggers; return payloads for the latest K sessions."""
        cap = self.config.update_every_n_sessions
        n = 0
        while not self.training_queue.empty() and n < cap:
            try:
                self.training_queue.get_nowait()
                self.training_queue.task_done()
                n += 1
            except asyncio.QueueEmpty:
                break

        k = self.config.training_window_sessions
        selected = self._session_history[-k:]
        orders = ", ".join(f"{o}:{sid or '?'}" for o, sid, _ in selected)
        log.info(
            "Training window: using %d session(s) from history=%d (K=%d) orders=[%s]",
            len(selected),
            len(self._session_history),
            k,
            orders,
        )
        return [data for _o, _sid, data in selected]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tinker proxy with rolling training window")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    if not args.config.endswith(".yaml"):
        raise SystemExit("Config file must be a YAML file")

    config = WindowConfig.from_yaml(args.config)
    server = WindowServer(config)
    app = server._build_app()
    uvicorn.run(app, host=config.proxy_host, port=config.proxy_port)
