"""Model-scoped state and clients for the Tinker proxy server.

This module owns everything about *which* model we're serving and training:

- The registry of slugs -> Tinker model paths (base models + produced checkpoints).
- Renderer / base-model resolution for a given model path.
- Slug arithmetic for naming new checkpoints.
- The Tinker SDK service and training clients, with per-base-model caching
  plus optional eager pre-warming at server start.
- The inference-probe loop that waits for a newly produced checkpoint to
  become servable before the proxy switches over.
- SSE-style broadcasting of model-update events to subscribers.
- Persistence of the model-scoped subset of ``state.json``.

The goal is to keep ``server.py`` focused on HTTP routing + queue plumbing
and route every "what model are we on?" question through a single object.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal

import tinker

from bridge_inference_client import BridgeInferenceClient
from trainer import Trainer

log = logging.getLogger(__name__)


# TODO: Reduce this list to only the models we want to support
TINKER_SUPPORTED_MODELS: list[str] = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-V3.1-Base",
    "moonshotai/Kimi-K2-Thinking",
    "moonshotai/Kimi-K2.5",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-4B",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
]


TINKER_MODEL_TO_RENDERER_NAME: dict[str, str] = {
    "meta-llama/Llama-3.2-1B": "llama3",
    "meta-llama/Llama-3.2-3B": "llama3",
    "meta-llama/Llama-3.1-8B": "llama3",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3",
    "meta-llama/Llama-3.1-70B": "llama3",
    "meta-llama/Llama-3.3-70B-Instruct": "llama3",
    "deepseek-ai/DeepSeek-V3.1": "deepseekv3_thinking",
    "deepseek-ai/DeepSeek-V3.1-Base": "deepseekv3_thinking",
    "moonshotai/Kimi-K2-Thinking": "kimi_k2",
    "moonshotai/Kimi-K2.5": "kimi_k25",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "qwen3_instruct",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "qwen3_instruct",
    "Qwen/Qwen3-30B-A3B": "qwen3",
    "Qwen/Qwen3-30B-A3B-Base": "qwen3",
    "Qwen/Qwen3-32B": "qwen3",
    "Qwen/Qwen3-8B": "qwen3",
    "Qwen/Qwen3-8B-Base": "qwen3",
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3_instruct",
    "Qwen/Qwen3-VL-235B-A22B-Instruct": "qwen3_vl_instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "qwen3_vl_instruct",
    "Qwen/Qwen3.5-397B-A17B": "qwen3_5",
    "Qwen/Qwen3.5-35B-A3B": "qwen3_5",
    "Qwen/Qwen3.6-35B-A3B": "qwen3_5",
    "Qwen/Qwen3.5-27B": "qwen3_5",
    "Qwen/Qwen3.5-4B": "qwen3_5",
    "openai/gpt-oss-120b": "gpt_oss_high_reasoning",
    "openai/gpt-oss-20b": "gpt_oss_low_reasoning",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "nemotron3",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nemotron3",
}


@dataclass
class ModelUpdate:
    """A structured record of a successful training round.

    Mirrors the payload that gets broadcast to SSE subscribers and persisted
    as ``latest_update`` in ``state.json``. Using a dataclass keeps the
    shape honest across server.py, the Electron client, and state files.
    """

    slug: str
    model_path: str
    base_model: str | None
    renderer_name: str | None
    mode: str
    state_path: str | None = None
    updated_at: float = field(default_factory=time.time)

    def to_event(self) -> dict:
        return asdict(self)


@dataclass
class _TrainingClientCacheEntry:
    client: "tinker.TrainingClient"
    load_checkpoint_path: str | None


class ModelManager:
    """Owns the model registry, Tinker clients, and model-scoped state.

    Concurrency model
    -----------------
    - ``model_ready`` gates inference while a checkpoint swap is in flight.
      Callers should ``await manager.model_ready.wait()`` before dispatching
      chat completions.
    - Training-client creation is serialized with a per-base-model lock so
      a burst of simultaneous "first hit for model X" requests coalesces
      into a single SDK call.
    - SSE broadcasts use non-blocking ``put_nowait`` so a slow subscriber
      can't stall training.
    """

    def __init__(
        self,
        *,
        tinker_api_key: str,
        tinker_base_url: str,
        state_path: str,
        mode: Literal["dpo", "reinforce", "opd"],
        lora_rank: int = 32,
        preload_model: str | None = None,
        supported_models: Iterable[str] = TINKER_SUPPORTED_MODELS,
        model_to_renderer: dict[str, str] | None = None,
    ) -> None:
        if not tinker_api_key:
            raise RuntimeError("tinker_api_key is required")

        self._tinker_api_key = tinker_api_key
        self._inference_base_url = tinker_base_url
        self._state_path = state_path
        self._mode = mode
        self._lora_rank = lora_rank
        self._preload_model = preload_model
        self._supported_models = list(supported_models)
        self._model_to_renderer = dict(model_to_renderer or TINKER_MODEL_TO_RENDERER_NAME)

        self._inference_client = BridgeInferenceClient(
            base_url=self._inference_base_url,
            api_key=tinker_api_key,
            resolve_renderer=self.resolve_renderer,
        )
        # Lazily built on first training-client request so unit tests and
        # dry-run usage don't force a network hop.
        self._service_client: tinker.ServiceClient | None = None

        # slug -> model path (either "org/Model" for a base model or
        # "tinker://..." for a saved checkpoint).
        self.models: dict[str, str] = {}
        # sampler checkpoint path -> training state checkpoint path. Inference
        # uses sampler paths; create_training_client_from_state needs state paths.
        self.training_state_paths: dict[str, str] = {}
        # The currently "active" model path (base or checkpoint). Updated
        # when a chat completion selects a slug and after a successful
        # training round swaps in a new checkpoint.
        self.model_path: str | None = None
        # Latest broadcasted update, kept so late SSE subscribers can catch up.
        self.latest_update: dict | None = None

        self.model_ready = asyncio.Event()
        self.model_ready.set()

        # base model id -> cached TrainingClient (+ checkpoint it was built from).
        self._training_clients: dict[str, _TrainingClientCacheEntry] = {}
        self._training_client_locks: dict[str, asyncio.Lock] = {}

        # base_model -> long-lived Trainer. Training mode is fixed for the
        # lifetime of a ModelManager (set via the ``mode`` constructor arg),
        # so we don't key on it here. Trainers sit next to the client cache
        # because each one borrows a TrainingClient from it; if the client
        # for a base model is invalidated, the matching trainer must go too.
        self._trainers: dict[str, "Trainer"] = {}
        self._trainer_locks: dict[str, asyncio.Lock] = {}

        self._subscribers: set[asyncio.Queue] = set()

        self._load_state()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Hook for FastAPI ``lifespan`` startup.

        Kicks off background pre-warming for the preload model if one is
        configured. We intentionally don't await the warmup here so HTTP
        startup is never blocked on Tinker provisioning.
        """
        if self._preload_model:
            base_model = self._preload_base_model()
            if base_model is not None:
                log.info("Pre-warming training client for %s", base_model)
                asyncio.create_task(self._warmup_training_client_bg(base_model))

    async def aclose(self) -> None:
        """Hook for FastAPI ``lifespan`` shutdown. Persists state."""
        for base_model, trainer in list(self._trainers.items()):
            try:
                await trainer.stop()
            except NotImplementedError:
                # Stub trainers (e.g. REINFORCE/OPD today) don't override stop.
                pass
            except Exception:
                log.exception("Trainer stop failed: base=%s", base_model)
        self._trainers.clear()
        self._trainer_locks.clear()
        self.save_state()

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        """The fixed training mode for this manager (``dpo``/``reinforce``/``opd``)."""
        return self._mode

    # ------------------------------------------------------------------
    # Inference client (OpenAI-compatible, pointed at Tinker)
    # ------------------------------------------------------------------
    @property
    def inference_client(self) -> BridgeInferenceClient:
        return self._inference_client

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------
    def list_slugs(self) -> list[str]:
        return sorted(self.models.keys())

    def has_slug(self, slug: str) -> bool:
        return slug in self.models

    def resolve_slug(self, slug: str) -> str:
        try:
            return self.models[slug]
        except KeyError as exc:
            raise KeyError(f"Unknown model slug: {slug}") from exc

    def set_active(self, model_path: str) -> None:
        """Record ``model_path`` as the current active model.

        Does not touch ``model_ready``; callers that need to gate inference
        should manage that event themselves (see ``update_model_path``).
        """
        self.model_path = model_path

    def register_checkpoint(self, slug: str, model_path: str, state_path: str | None = None) -> None:
        """Add a newly produced sampler checkpoint to the registry and persist."""
        self.models[slug] = model_path
        if state_path is not None:
            self.training_state_paths[model_path] = state_path
        self.save_state()

    def training_state_path_for(self, model_path: str) -> str | None:
        """Return the resumable training state path for a sampler checkpoint."""
        state_path = self.training_state_paths.get(model_path)
        if state_path is not None:
            return state_path

        # Backfill checkpoints saved before state paths were persisted. Tinker
        # uses sibling namespaces for resumable state and sampler weights.
        if "/sampler_weights/" in model_path:
            return model_path.replace("/sampler_weights/", "/weights/", 1)

        return None

    def next_slug(self, mode: str) -> str:
        """Compute the next available ``<base>-<mode>-vN`` slug.

        Scans existing slugs to pick ``N`` so successive training rounds
        get monotonically increasing versions even across restarts.
        """
        if self.model_path and self.model_path.startswith("tinker://"):
            base_model_name, _ = self.resolve_renderer(self.model_path)
            base_short = base_model_name.split("/")[-1]
        elif self.model_path:
            base_short = self.model_path.split("/")[-1]
        else:
            base_short = "model"

        stem = f"{base_short}-{mode}"
        pattern = re.compile(rf"^{re.escape(stem)}-v(\d+)$")
        max_version = 0
        for existing in self.models.keys():
            m = pattern.match(existing)
            if m:
                max_version = max(max_version, int(m.group(1)))

        version = max_version + 1
        slug = f"{stem}-v{version}"
        while slug in self.models:
            version += 1
            slug = f"{stem}-v{version}"
        return slug

    # ------------------------------------------------------------------
    # Renderer / base-model resolution
    # ------------------------------------------------------------------
    def resolve_renderer(self, model_path: str) -> tuple[str, str]:
        """Return ``(base_model_name, renderer_name)`` for a path.

        Accepts either a plain ``"org/Model"`` base id or a
        ``"tinker://.../<something>_<BaseModelShortName>_<...>"`` checkpoint
        path and recovers the full base model id + its renderer registration.

        The checkpoint-name schema is owned by the trainers (see
        ``trainer.py``) and has changed over time (``final_<MODE>_<base>_<ts>``
        vs. ``round_<N>_<mode>_<base>_<ts>``), so we don't hard-code a token
        index. Instead, we scan underscore-delimited tokens for the first one
        that matches a known supported base-model short name. None of the
        registered short names contain underscores, so this is unambiguous.
        """
        if model_path.startswith("tinker://"):
            tail = model_path.split("/")[-1]
            known = {
                full.split("/")[-1]: full for full in self._model_to_renderer
            }
            base_model: str | None = None
            for token in tail.split("_"):
                full = known.get(token)
                if full is not None:
                    base_model = full
                    break
            if base_model is None:
                raise ValueError(
                    f"Could not identify a supported base model in checkpoint "
                    f"path {model_path!r}"
                )
        else:
            base_model = model_path

        try:
            renderer_name = self._model_to_renderer[base_model]
        except KeyError as exc:
            raise KeyError(
                f"No renderer registered for base model {base_model!r} "
                f"(derived from {model_path!r})"
            ) from exc
        return base_model, renderer_name

    # ------------------------------------------------------------------
    # Training client cache
    # ------------------------------------------------------------------
    def get_service_client(self) -> tinker.ServiceClient:
        """Return the lazily-built shared ``tinker.ServiceClient``.

        Exposed publicly so trainer builders (see ``get_trainer``) can
        reuse the same SDK connection rather than constructing their own.
        """
        if self._service_client is None:
            self._service_client = tinker.ServiceClient()
        return self._service_client

    # Backwards-compatible alias; internal callers used the underscore name.
    _get_service_client = get_service_client

    async def get_training_client(
        self,
        base_model: str,
        *,
        load_checkpoint_path: str | None = None,
        user_metadata: dict[str, str] | None = None,
    ) -> tinker.TrainingClient:
        """Return a ``TrainingClient`` for ``base_model``, creating on first use.

        The cache is keyed by ``base_model``. If a cached client was built
        for a different ``load_checkpoint_path`` than requested, it's
        invalidated and recreated -- this keeps semantics obvious at the
        cost of dropping Tinker-side state on checkpoint changes.

        Concurrent first-hits for the same base model coalesce behind a
        per-key lock so we only hit the SDK once.
        """
        lock = self._training_client_locks.setdefault(base_model, asyncio.Lock())
        async with lock:
            entry = self._training_clients.get(base_model)
            if entry is not None and entry.load_checkpoint_path == load_checkpoint_path:
                return entry.client
            if entry is not None:
                log.info(
                    "Invalidating cached training client for %s "
                    "(checkpoint %s -> %s)",
                    base_model, entry.load_checkpoint_path, load_checkpoint_path,
                )
            client = await self._create_training_client(
                base_model=base_model,
                load_checkpoint_path=load_checkpoint_path,
                user_metadata=user_metadata,
            )
            self._training_clients[base_model] = _TrainingClientCacheEntry(
                client=client,
                load_checkpoint_path=load_checkpoint_path,
            )
            return client

    def invalidate_training_client(self, base_model: str) -> None:
        """Drop the cached ``TrainingClient`` and the trainer that borrowed it.

        A trainer holds a reference to the client; leaving it in place after
        the cache is cleared would let new callers reconnect to a freshly
        built client while the existing trainer kept driving the old one.
        Evict both together.
        """
        self._training_clients.pop(base_model, None)
        self._trainers.pop(base_model, None)
        self._trainer_locks.pop(base_model, None)

    # ------------------------------------------------------------------
    # Trainer registry
    # ------------------------------------------------------------------
    async def get_trainer(
        self,
        *,
        base_model: str,
        build: Callable[
            [tinker.TrainingClient, tinker.ServiceClient],
            "Trainer",
        ],
        load_checkpoint_path: str | None = None,
    ) -> "Trainer":
        """Return a long-lived ``Trainer`` for ``base_model``.

        Trainers are cached because they carry step counters, optimizer
        state, loggers, and a live ``TrainingClient`` across many
        ``do_update`` rounds. The training mode is fixed by ``self._mode``
        (set once in the constructor), so the cache key is just the base
        model id. Concurrent first-hits coalesce behind a per-key lock,
        mirroring ``get_training_client``.

        ``build`` is a callback that receives the (cached) training and
        service clients and returns a fully-constructed trainer. Passing
        the clients in rather than letting the trainer construct them
        keeps a single source of truth for Tinker connections.
        """
        lock = self._trainer_locks.setdefault(base_model, asyncio.Lock())
        async with lock:
            existing = self._trainers.get(base_model)
            if existing is not None:
                return existing
            training_client = await self.get_training_client(
                base_model,
                load_checkpoint_path=load_checkpoint_path,
            )
            service_client = self.get_service_client()
            trainer = build(training_client, service_client)
            self._trainers[base_model] = trainer
            return trainer

    async def invalidate_trainer(self, base_model: str) -> None:
        """Drop a single trainer, calling ``stop`` if implemented."""
        trainer = self._trainers.pop(base_model, None)
        self._trainer_locks.pop(base_model, None)
        if trainer is None:
            return
        try:
            await trainer.stop()
        except NotImplementedError:
            pass
        except Exception:
            log.exception("Trainer stop failed during invalidate: base=%s", base_model)

    async def _create_training_client(
        self,
        *,
        base_model: str,
        load_checkpoint_path: str | None,
        user_metadata: dict[str, str] | None,
    ) -> tinker.TrainingClient:
        service_client = self._get_service_client()
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()

        if load_checkpoint_path is not None:
            log.info(
                "Creating TrainingClient from state: base=%s checkpoint=%s",
                base_model, load_checkpoint_path,
            )
            client = await loop.run_in_executor(
                None,
                lambda: service_client.create_training_client_from_state(
                    load_checkpoint_path,
                    user_metadata=user_metadata,
                ),
            )
        else:
            log.info(
                "Creating LoRA TrainingClient: base=%s rank=%d",
                base_model, self._lora_rank,
            )
            client = await loop.run_in_executor(
                None,
                lambda: service_client.create_lora_training_client(
                    base_model=base_model,
                    rank=self._lora_rank,
                    user_metadata=user_metadata,
                ),
            )

        log.info(
            "TrainingClient ready: base=%s elapsed=%.2fs",
            base_model, time.monotonic() - t0,
        )
        return client

    async def _warmup_training_client_bg(self, base_model: str) -> None:
        try:
            await self.get_training_client(base_model)
        except Exception:
            log.exception("Pre-warm failed for %s", base_model)

    def _preload_base_model(self) -> str | None:
        """Resolve ``preload_model`` to a concrete base model.

        Accepts either a known slug from the registry or a bare base model
        id. Returns ``None`` with a warning if neither matches, rather
        than failing server startup for a misconfigured preload.
        """
        candidate = self._preload_model
        if not candidate:
            return None
        if candidate in self.models:
            path = self.models[candidate]
            try:
                base_model, _ = self.resolve_renderer(path)
                return base_model
            except Exception:
                log.exception("Cannot resolve base model for preload slug %s", candidate)
                return None
        if candidate in self._model_to_renderer:
            return candidate
        log.warning(
            "preload_model=%r is neither a known slug nor a supported base model; skipping warmup",
            candidate,
        )
        return None

    # ------------------------------------------------------------------
    # Active-model probe
    # ------------------------------------------------------------------
    async def update_model_path(
        self,
        model_path: str,
        *,
        timeout: float = 120.0,
        interval: float = 2.0,
    ) -> None:
        """Block until ``model_path`` serves a trivial chat completion.

        Used after a training round to wait out the Tinker-side deploy
        before flipping the active model. Raises ``TimeoutError`` if the
        checkpoint never becomes ready, leaving ``self.model_path``
        unchanged so the server can keep serving the previous model.
        """
        deadline = time.monotonic() + timeout
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                await self._inference_client.chat.completions.create(
                    model=model_path,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                )
                self.model_path = model_path
                log.info(
                    "Model path updated: %s (ready after %d probe(s))",
                    model_path, attempt,
                )
                return
            except Exception as exc:
                log.debug(
                    "Model probe attempt %d for %s failed: %s",
                    attempt, model_path, exc,
                )
                await asyncio.sleep(interval)
        raise TimeoutError(
            f"Model {model_path} not ready after {timeout}s ({attempt} probes)"
        )

    # ------------------------------------------------------------------
    # SSE broadcast
    # ------------------------------------------------------------------
    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        log.info("SSE subscriber connected: total=%d", len(self._subscribers))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)
        log.info("SSE subscriber disconnected: total=%d", len(self._subscribers))

    def broadcast(self, update: ModelUpdate | dict) -> None:
        event = update.to_event() if isinstance(update, ModelUpdate) else update
        self.latest_update = event
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except Exception:
                # A broken subscriber must never stall training.
                log.exception("Failed to enqueue model update for subscriber")

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        """Seed the registry with supported base models, then overlay state.

        The supported-models seed runs unconditionally so new entries added
        to ``TINKER_SUPPORTED_MODELS`` show up even when ``state.json``
        pre-dates them. On-disk entries win for slugs that already exist
        (they may point at trained checkpoints rather than base models).
        """
        for model_name in self._supported_models:
            self.models[model_name.split("/")[-1]] = model_name

        if not os.path.exists(self._state_path):
            return

        try:
            with open(self._state_path) as f:
                state = json.load(f)
        except Exception:
            log.exception("Failed to load state from %s", self._state_path)
            return

        self.models.update(state.get("models", {}))
        self.training_state_paths.update(state.get("training_state_paths", {}))
        self.latest_update = state.get("latest_update")
        if self.latest_update is not None:
            model_path = self.latest_update.get("model_path")
            state_path = self.latest_update.get("state_path")
            if model_path and state_path:
                self.training_state_paths.setdefault(model_path, state_path)
        self.model_path = state.get("model_path")
        log.info(
            "Loaded state from %s: %d model slugs, %d training states, model_path=%s, latest_update=%s",
            self._state_path,
            len(self.models),
            len(self.training_state_paths),
            self.model_path,
            "present" if self.latest_update is not None else "none",
        )

    def save_state(self) -> None:
        Path(self._state_path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "models": self.models,
            "training_state_paths": self.training_state_paths,
            "latest_update": self.latest_update,
            "model_path": self.model_path,
        }
        try:
            with open(self._state_path, "w") as f:
                json.dump(state, f)
        except Exception:
            log.exception("Failed to save state to %s", self._state_path)
            return
        log.info(
            "Saved state to %s: %d model slugs, %d training states, model_path=%s, latest_update=%s",
            self._state_path,
            len(self.models),
            len(self.training_state_paths),
            self.model_path,
            "present" if self.latest_update is not None else "none",
        )
