"""Tinker proxy with capped replay over all previously seen sessions.

This variant keeps the same HTTP API as ``server.py`` and ``server_window.py``.
It stores submitted sessions in arrival order, then each training round uses
the full eligible history, where eligibility means the session's replay count
is still below ``training_session_max_uses``. Every eligible session included
in a round has its use count incremented when the round is drained for training.

Run::

    python server_capped_replay.py --config config-dpo-capped-replay.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, fields
from pathlib import Path

import uvicorn
import yaml

from server import Config, Server

log = logging.getLogger(__name__)

_TRAIN_TRIGGER = object()


@dataclass
class CappedReplayConfig(Config):
    """``Config`` plus capped replay controls."""

    training_min_sessions: int | None = None
    training_session_max_uses: int = 4

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CappedReplayConfig":
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        # Backward compatibility for older configs.
        if "training_min_sessions" not in raw and "training_window_min_sessions" in raw:
            raw["training_min_sessions"] = raw["training_window_min_sessions"]
        valid_keys = {fld.name for fld in fields(cls)}
        unknown = set(raw) - valid_keys
        if unknown:
            log.warning("Ignoring unknown config keys in %s: %s", path, ", ".join(sorted(unknown)))
        return cls(**{k: v for k, v in raw.items() if k in valid_keys})


class CappedReplayServer(Server):
    """Train on all eligible previously seen sessions."""

    def __init__(self, config: CappedReplayConfig):
        if config.training_session_max_uses < 1:
            raise ValueError("training_session_max_uses must be >= 1")
        min_sessions = 1 if config.training_min_sessions is None else config.training_min_sessions
        if min_sessions < 1:
            raise ValueError("training_min_sessions must be >= 1")
        super().__init__(config)
        self._min_training_sessions = min_sessions
        self._session_history: list[tuple[int, str | None, dict]] = []
        self._session_use_counts: dict[int, int] = {}
        self._next_order = 0

    async def _process_sessions(self) -> None:
        while True:
            session_data = await self.sessions_queue.get()
            try:
                if self.config.mode not in ("dpo", "opd", "reinforce"):
                    raise ValueError(
                        f"Unknown training mode: {self.config.mode} "
                        "(expected one of 'dpo', 'opd', 'reinforce')"
                    )

                order = self._next_order
                self._next_order += 1
                session_id = session_data.get("uuid")
                self._session_history.append((order, session_id, session_data))
                self._session_use_counts[order] = 0

                unit_count = len(session_data.get("task_units", []))
                if unit_count == 0:
                    unit_count = len(session_data.get("learning_units", []))

                history_len = len(self._session_history)
                if history_len < self._min_training_sessions:
                    log.info(
                        "Session processed: order=%d id=%s mode=%s units=%d "
                        "history=%d min_train=%d — warmup, skipping training",
                        order,
                        session_id,
                        self.config.mode,
                        unit_count,
                        history_len,
                        self._min_training_sessions,
                    )
                    continue

                await self.training_queue.put(_TRAIN_TRIGGER)
                pending = self.training_queue.qsize()
                log.info(
                    "Session processed: order=%d id=%s mode=%s units=%d "
                    "history=%d pending_triggers=%d/%d",
                    order,
                    session_id,
                    self.config.mode,
                    unit_count,
                    history_len,
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
        """Consume triggers and return all currently eligible payloads."""
        cap = self.config.update_every_n_sessions
        n = 0
        while not self.training_queue.empty() and n < cap:
            try:
                self.training_queue.get_nowait()
                self.training_queue.task_done()
                n += 1
            except asyncio.QueueEmpty:
                break

        max_uses = self.config.training_session_max_uses
        selected = [
            record
            for record in self._session_history
            if self._session_use_counts.get(record[0], 0) < max_uses
        ]

        for order, _sid, _data in selected:
            self._session_use_counts[order] = self._session_use_counts.get(order, 0) + 1

        orders = ", ".join(
            f"{order}:{sid or '?'}#{self._session_use_counts.get(order, 0)}"
            for order, sid, _ in selected
        )
        log.info(
            "Capped replay training: using %d eligible session(s), history=%d "
            "max_uses=%d orders=[%s]",
            len(selected),
            len(self._session_history),
            max_uses,
            orders,
        )
        return [data for _order, _sid, data in selected]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tinker proxy with capped replay over all eligible sessions"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    if not args.config.endswith(".yaml"):
        raise SystemExit("Config file must be a YAML file")

    config = CappedReplayConfig.from_yaml(args.config)
    server = CappedReplayServer(config)
    app = server._build_app()
    uvicorn.run(app, host=config.proxy_host, port=config.proxy_port)


if __name__ == "__main__":
    main()
