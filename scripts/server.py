import json
import os
import re
import tempfile
import yaml
import logging
import asyncio
import argparse
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import uvicorn
from openai import AsyncOpenAI
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from contextlib import asynccontextmanager

import session_export_common as s
from export_opd_data import _session as _opd_session
from export_dpo_data import _session as _dpo_session
from export_reinforce_data import _session as _reinforce_session
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_opd import Config as OPDConfig, main as _opd_train
from tinker_dpo import Config as DPOConfig, main as _dpo_train
from tinker_reinforce import Config as ReinforceConfig, main as _reinforce_train
from tinker_formatter import DPODataBuilder, OPDSDFTDataset, ReinforceDataBuilder
from toolcall_parser import *

from dotenv import load_dotenv
load_dotenv()


TINKER_SUPPORTED_MODELS = [
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
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-4B",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
]


TINKER_MODEL_TO_RENDERER_NAME = {
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
    "Qwen/Qwen3.5-27B": "qwen3_5",
    "Qwen/Qwen3.5-4B": "qwen3_5",
    "openai/gpt-oss-120b": "gpt_oss_high_reasoning",
    "openai/gpt-oss-20b": "gpt_oss_low_reasoning",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "nemotron3",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nemotron3",
}


@dataclass
class Config:
    # -- Mode --
    mode: Literal["dpo", "opd", "reinforce"] = "reinforce"
    
    # -- Inference --
    temperature: float = 0.0
    logprobs: bool = False
    
    # -- Update --
    update_every_n_sessions: int = 1 # update the model after n new sessions are processed
    
    # -- Training --
    lora_rank: int = 32
    dry_run: bool = False # skip training if enabled
    learning_rate: float = 1e-5
    num_epochs: int = 1
    batch_size: int = 1
    lr_schedule: str = "linear"
    max_steps: int | None = None

    # -- DPO-specific --
    dpo_beta: float = 0.1

    # -- OPD-specific --
    opd_max_tokens: int = 2048
    opd_temperature: float = 1.0
    opd_topk: int = 20
    opd_max_context_length: int = 32768

    # -- REINFORCE-specific --
    reward_alpha: float = 0.05
    initial_baseline: float = 0.0

    # -- Logging --
    wandb_project: str | None = None
    wandb_name: str | None = None

    # -- Proxy Server --
    proxy_host: str = "localhost"
    proxy_port: int = 8000
    state_path: str = "state.json"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        valid_keys = {fld.name for fld in fields(cls)}
        unknown = set(raw) - valid_keys
        if unknown:
            log.warning("Ignoring unknown config keys in %s: %s", path, ", ".join(sorted(unknown)))
        return cls(**{k: v for k, v in raw.items() if k in valid_keys})


@asynccontextmanager
async def lifespan(app: FastAPI):
    owner = app.state.owner    
    t1 = asyncio.create_task(owner._process_sessions())
    t2 = asyncio.create_task(owner._training_consumer())
    yield
    t1.cancel()
    t2.cancel()
    for t in (t1, t2):
        try:
            await t
        except asyncio.CancelledError:
            pass
    owner._save_state()
  

class Server:
    _TITLE = "Agent Cowork Tinker Proxy"
    _TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    _API_KEY = os.getenv("TINKER_API_KEY")
    
    def __init__(
        self,
        config: Config
    ):
        if not self._API_KEY:
            raise RuntimeError("TINKER_API_KEY environment variable is required")
        self.client = AsyncOpenAI(
            base_url=self._TINKER_BASE_URL,
            api_key=self._API_KEY,
        )
        
        self.config = config
        
        self.sessions_queue = asyncio.Queue()
        self.training_queue = asyncio.Queue()
        self.training_event = asyncio.Event()
        self.training_lock = asyncio.Lock()
        
        # a list of all available models
        # format: 
        # 1. base model
        # 2. checkpoints: maps from a customized name to the checkpoint path
        self.models: dict[str, str] = {}
            
        # initialize variables
        self.model_path = None
        
        # event to signal that the model is ready to be used
        self.model_ready = asyncio.Event()
        self.model_ready.set()
        
        # latest successful training update, pushed to clients via SSE
        # None until the first training round completes
        self.latest_update: dict | None = None
        # subscribers for /v1/tinker/events; each connection owns an asyncio.Queue
        self._update_subscribers: set[asyncio.Queue] = set()

        # Restore persisted state last so it overrides the defaults above
        # rather than being clobbered by them.
        self._load_state()
        
    def _get_provider(self, model_name_suffix: str) -> str:
        if "Llama" in model_name_suffix:
            return "meta-llama"
        elif "DeepSeek" in model_name_suffix:
            return "deepseek-ai"
        elif "Kimi" in model_name_suffix:
            return "moonshotai"
        elif "Qwen" in model_name_suffix:
            return "Qwen"
        elif "gpt" in model_name_suffix:
            return "openai"
        elif "NVIDIA" in model_name_suffix:
            return "nvidia"
        else:
            raise ValueError(f"Unknown provider: {model_name_suffix}")
        
    def _get_model_renderer_name(self, model_path: str) -> str:
        if model_path.startswith("tinker://"):
            model_name_suffix = model_path.split("/")[-1].split("_")[2]
            model_name = self._get_provider(model_name_suffix) + "/" + model_name_suffix
        else:
            model_name = model_path
        renderer_name = TINKER_MODEL_TO_RENDERER_NAME[model_name]
        
        return model_name, renderer_name
        
    def _next_model_slug(self) -> str:
        if self.model_path and self.model_path.startswith("tinker://"):
            base_model_name, _ = self._get_model_renderer_name(self.model_path)
            base_short = base_model_name.split("/")[-1]
        elif self.model_path:
            base_short = self.model_path.split("/")[-1]
        else:
            base_short = "model"

        stem = f"{base_short}-{self.config.mode}"

        pattern = re.compile(rf"^{re.escape(stem)}-v(\d+)$")
        max_version = 0
        for existing in self.models.keys():
            m = pattern.match(existing)
            if m:
                max_version = max(max_version, int(m.group(1)))

        slug = f"{stem}-v{max_version + 1}"
        while slug in self.models:
            max_version += 1
            slug = f"{stem}-v{max_version + 1}"
        return slug
    
    def _broadcast_model_update(self, event: dict) -> None:
        """Fan out a model-update event to every active SSE subscriber.

        Subscribers own unbounded queues; we swallow errors so a slow/buggy
        subscriber can't block training.
        """
        self.latest_update = event
        for q in list(self._update_subscribers):
            try:
                q.put_nowait(event)
            except Exception:
                log.exception("Failed to enqueue model update for subscriber")

    def _build_app(self) -> FastAPI:
        app = FastAPI(title=self._TITLE, lifespan=lifespan)
        app.state.owner = self

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}
        
        @app.post("/session")
        async def _handle_session(request: Request):
            body = await request.json()
            
            session_id = body.get("uuid")
            name = body.get("name", "")
            trajectory = body.get("trajectory")
            
            if not isinstance(trajectory, list) or len(trajectory) == 0:
                raise HTTPException(status_code=400, detail="trajectory must be a non-empty list")

            
            await self.sessions_queue.put({
                "uuid": session_id,
                "name": name,
                "trajectory": trajectory,
            })
            
            log.info("Session enqueued: id=%s name=%r steps=%d queue_depth=%d",
                     session_id, name, len(trajectory), self.sessions_queue.qsize())
            return {"ok": True, "session_id": session_id}
        
        @app.get("/v1/models")
        async def _get_models(request: Request):
            data = []
            for model_name in self.models.keys():
                data.append({
                    "id": model_name,
                    "object": "model",
                    "created": None,
                    "owned_by": "agent-cowork", # TODO: update when we have a proper name for the app
                })
            data.sort(key=lambda x: x["id"])
                
            return JSONResponse({"object": "list", "data": data})
        
        @app.get("/v1/tinker/current")
        async def _get_tinker_current():
            """Return the most recent model update (or 204 if none yet).

            Used by the Electron main process to reconcile on startup/reconnect.
            """
            if self.latest_update is None:
                return Response(status_code=204)
            return JSONResponse(self.latest_update)
        
        @app.get("/v1/tinker/events")
        async def _stream_tinker_events(request: Request):
            """Server-Sent Events stream of model-update notifications.

            On connect we immediately emit the current latest_update (if any)
            so late subscribers converge without needing a separate GET.
            Heartbeat comments keep proxies/load balancers from closing the
            connection while training is idle.
            """
            queue: asyncio.Queue = asyncio.Queue()
            self._update_subscribers.add(queue)
            log.info("SSE subscriber connected: total=%d", len(self._update_subscribers))

            async def event_stream():
                try:
                    if self.latest_update is not None:
                        yield f"event: model-update\ndata: {json.dumps(self.latest_update)}\n\n"
                    while True:
                        if await request.is_disconnected():
                            break
                        try:
                            event = await asyncio.wait_for(queue.get(), timeout=15.0)
                        except asyncio.TimeoutError:
                            yield ": heartbeat\n\n"
                            continue
                        yield f"event: model-update\ndata: {json.dumps(event)}\n\n"
                finally:
                    self._update_subscribers.discard(queue)
                    log.info("SSE subscriber disconnected: total=%d", len(self._update_subscribers))

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )
        
        @app.post("/v1/chat/completions")
        async def _handle_chat_completions(request: Request):
            # block until the model is ready to be used
            await self.model_ready.wait()
            
            body = await request.json()
            
            messages = body.pop("messages", [])
            stream = body.pop("stream", True)
            model_name = body.pop("model", None)
            if model_name is None:
                raise HTTPException(status_code=400, detail="model is required")
            if model_name not in self.models:
                raise HTTPException(status_code=400, detail=f"model {model_name} not found")
            
            # switch the model to be trained to the selected model
            self.model_path = self.models[model_name]
            log.info("Switching model to: %s", self.model_path)
            
            log.info("Chat completion request: model=%s stream=%s messages=%d",
                     self.model_path, stream, len(messages))
            
            # Override inference parameters
            body["temperature"] = self.config.temperature
            body["logprobs"] = self.config.logprobs
            
            if stream:
                # Derive the renderer from the resolved model, not a nonexistent
                # config field. Falls back to pass-through if the base model has
                # no registered rewriter.
                try:
                    _, renderer_name = self._get_model_renderer_name(self.model_path)
                except Exception:
                    log.exception("Failed to resolve renderer for %s; streaming without rewrite", self.model_path)
                    renderer_name = None
                rewrite = func_mapping.get(renderer_name) if renderer_name else None
                
                async def generate():
                    chunks = []
                    state = {}
                    
                    response = await self.client.chat.completions.create(
                        model=self.model_path,
                        messages=messages,
                        stream=True,
                        **body,
                    )
                    async for chunk in response:
                        log.info("Received chat completion chunk: %s", json.dumps(chunk.model_dump(), indent=4))
                        chunk_dict = chunk.model_dump()
                        if rewrite:
                            rewritten = rewrite(chunk_dict, state)
                        else:
                            rewritten = chunk_dict
                        chunks.append(chunk_dict)
                        if rewritten is not None:
                            yield f"data: {json.dumps(rewritten)}\n\n"
                    yield "data: [DONE]\n\n"
                    log.info("Streamed chat completion chunks: %s", json.dumps(chunks, indent=4))
                return StreamingResponse(generate(), media_type="text/event-stream")

            response = await self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                **body,
            )
            log.info("Chat completion response: %s", response.model_dump_json(indent=4))
            return JSONResponse(response.model_dump())
        
        return app
    
    async def _process_sessions(self):
        loop = asyncio.get_running_loop()
        while True:
            session_data = await self.sessions_queue.get()
            try:
                for session_blob in s.blobs(session_data):
                    if self.config.mode == "dpo":
                        data = await loop.run_in_executor(None, _dpo_session, session_blob)
                    elif self.config.mode == "opd":
                        data = await loop.run_in_executor(None, _opd_session, session_blob)
                    elif self.config.mode == "reinforce":
                        data = await loop.run_in_executor(None, _reinforce_session, session_blob)
                    else:
                        raise ValueError(f"Unknown training mode: {self.config.mode} (expected one of 'dpo', 'opd', 'reinforce')")
                
                    if data:
                        await self.training_queue.put(data)
                        log.info("Session processed: id=%s mode=%s units=%d training_queue=%d/%d",
                                session_data.get('uuid'), self.config.mode,
                                len(data.get('learning_units', [])),
                                self.training_queue.qsize(), self.config.update_every_n_sessions)
                    
                    # trigger training if the queue has reached the update threshold
                    if self.training_queue.qsize() >= self.config.update_every_n_sessions:
                        self.training_event.set()

            except Exception:
                log.exception("Session processing failed: id=%s mode=%s",
                              session_data.get('uuid'), self.config.mode)
                
            finally:
                self.sessions_queue.task_done()
    
    def _drain_queue(self) -> list:
        items = []
        while not self.training_queue.empty():
            try:
                items.append(self.training_queue.get_nowait())
                self.training_queue.task_done()
            except asyncio.QueueEmpty:
                break
        return items
    
    async def _training_consumer(self):
        while True:
            await self.training_event.wait()
            self.training_event.clear()

            async with self.training_lock:
                items = self._drain_queue()
                
                # Skip training if dry run is enabled, for the sake of testing other parts of the server
                if self.config.dry_run:
                    log.info("Dry run: skipping training for %d sessions", len(items))
                    continue
                
                if not items:
                    log.info("No sessions to train, skipping")
                    continue

                log.info("Training started: mode=%s sessions=%d current_model=%s",
                         self.config.mode, len(items), self.model_path)

                loop = asyncio.get_running_loop()
                try:
                    if self.config.mode == "dpo":
                        checkpoint = await loop.run_in_executor(None, self._run_dpo_training, items)
                    elif self.config.mode == "opd":
                        checkpoint = await loop.run_in_executor(None, self._run_opd_training, items)
                    elif self.config.mode == "reinforce":
                        checkpoint = await loop.run_in_executor(None, self._run_reinforce_training, items)
                    else:
                        raise ValueError(f"Invalid mode: {self.config.mode}")
                    
                    prev_model = self.model_path
                    
                    # update the model path to the new checkpoint
                    self.model_ready.clear()
                    try:
                        log.info("Updating model path to %s", checkpoint)
                        await self.update_model_path(checkpoint, timeout=600.0)
                        
                        # update the model name to the next model slug
                        new_model_name = self._next_model_slug()
                        self.models[new_model_name] = checkpoint
                        log.info("Model updated: name=%s path=%s", new_model_name, checkpoint)
                        
                        # Broadcast to SSE subscribers so the Electron app can
                        # atomically rotate the Tinker provider config onto the
                        # new slug without polling.
                        try:
                            base_model_name, renderer_name = self._get_model_renderer_name(checkpoint)
                        except Exception:
                            log.exception("Failed to resolve renderer for %s; broadcasting without renderer", checkpoint)
                            base_model_name, renderer_name = None, None
                        self._broadcast_model_update({
                            "slug": new_model_name,
                            "model_path": checkpoint,
                            "base_model": base_model_name,
                            "renderer_name": renderer_name,
                            "mode": self.config.mode,
                            "updated_at": time.time(),
                        })
                        self._save_state()
                        
                    except TimeoutError:
                        log.error("Checkpoint %s never became ready, keeping previous model %s",
                                checkpoint, prev_model)
                    finally:
                        self.model_ready.set()
                    
                    log.info("Training complete: mode=%s sessions=%d model=%s (was %s)",
                             self.config.mode, len(items), checkpoint, prev_model)
                except Exception:
                    log.exception("Training failed: mode=%s sessions=%d",
                                  self.config.mode, len(items))

            if self.training_queue.qsize() >= self.config.update_every_n_sessions:
                self.training_event.set()
                
    async def update_model_path(self, model_path: str, timeout: float = 120, interval: float = 2.0):
        deadline = time.monotonic() + timeout
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                await self.client.chat.completions.create(
                    model=model_path,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                )
                self.model_path = model_path
                log.info("Model path updated: %s (ready after %d probe(s))", model_path, attempt)
                return
            except Exception as e:
                log.debug("Model probe attempt %d for %s failed: %s", attempt, model_path, e)
                await asyncio.sleep(interval)
        raise TimeoutError(
            f"Model {model_path} not ready after {timeout}s ({attempt} probes)"
        )

    def _write_sessions_file(self, items: list) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({"sessions": items}, f)
        f.close()
        return f.name

    def _get_checkpoint_path(self, log_path: str) -> str:
        last = checkpoint_utils.get_last_checkpoint(log_path)
        if last is None or last.sampler_path is None:
            raise RuntimeError(f"No sampling checkpoint found in {log_path} after training")
        return last.sampler_path
    
    def _check_if_use_checkpoint(self) -> bool:
        # helper function to decide whether to start training from scratch or from the current model checkpoint
        return self.model_path.startswith("tinker://")

    def _run_dpo_training(self, items: list) -> str:
        # TODO: improve efficiency
        train_path = self._write_sessions_file(items)
        model_name, renderer_name = self._get_model_renderer_name(self.model_path)
        try:
            log_path = f"logs/tinker_dpo/{int(time.time())}"
            dataset_builder = DPODataBuilder(
                train_path=train_path,
                common_config=ChatDatasetBuilderCommonConfig(
                    model_name_for_tokenizer=model_name,
                    renderer_name=renderer_name,
                    max_length=None,
                    batch_size=self.config.batch_size,
                ),
            )
            config = DPOConfig(
                log_path=log_path,
                model_name=model_name,
                dataset_builder=dataset_builder,
                renderer_name=renderer_name,
                learning_rate=self.config.learning_rate,
                lr_schedule=self.config.lr_schedule,
                dpo_beta=self.config.dpo_beta,
                num_epochs=self.config.num_epochs,
                lora_rank=self.config.lora_rank,
                max_steps=self.config.max_steps,
                wandb_project=self.config.wandb_project,
                wandb_name=self.config.wandb_name,
                load_checkpoint_path=self.model_path if self._check_if_use_checkpoint() else None,
            )
            _dpo_train(config)
            return self._get_checkpoint_path(log_path)
        finally:
            os.unlink(train_path)

    def _run_opd_training(self, items: list) -> str:
        train_path = self._write_sessions_file(items)
        model_name, renderer_name = self._get_model_renderer_name(self.model_path)
        try:
            log_path = f"logs/tinker_opd/{int(time.time())}"
            tokenizer = get_tokenizer(model_name)
            renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
            sdft_dataset = OPDSDFTDataset.from_json(
                data_path=train_path,
                renderer=renderer,
                batch_size=self.config.batch_size,
            )
            config = OPDConfig(
                model_name=model_name,
                renderer_name=renderer_name,
                log_path=log_path,
                lora_rank=self.config.lora_rank,
                learning_rate=self.config.learning_rate,
                max_tokens=self.config.opd_max_tokens,
                temperature=self.config.opd_temperature,
                topk=self.config.opd_topk,
                max_context_length=self.config.opd_max_context_length,
                max_steps=self.config.max_steps,
                wandb_project=self.config.wandb_project,
                wandb_name=self.config.wandb_name,
                load_checkpoint_path=self.model_path if self._check_if_use_checkpoint() else None,
            )
            asyncio.run(_opd_train(config, sdft_dataset))
            return self._get_checkpoint_path(log_path)
        finally:
            os.unlink(train_path)

    def _run_reinforce_training(self, items: list) -> str:
        train_path = self._write_sessions_file(items)
        model_name, renderer_name = self._get_model_renderer_name(self.model_path)
        try:
            log_path = f"logs/tinker_reinforce/{int(time.time())}"
            dataset_builder = ReinforceDataBuilder(
                train_path=train_path,
                reward_alpha=self.config.reward_alpha,
                common_config=ChatDatasetBuilderCommonConfig(
                    model_name_for_tokenizer=model_name,
                    renderer_name=renderer_name,
                    max_length=None,
                    batch_size=self.config.batch_size,
                ),
            )
            config = ReinforceConfig(
                log_path=log_path,
                model_name=model_name,
                dataset_builder=dataset_builder,
                renderer_name=renderer_name,
                learning_rate=self.config.learning_rate,
                lr_schedule=self.config.lr_schedule,
                num_epochs=self.config.num_epochs,
                lora_rank=self.config.lora_rank,
                reward_alpha=self.config.reward_alpha,
                initial_baseline=self.config.initial_baseline,
                max_steps=self.config.max_steps,
                wandb_project=self.config.wandb_project,
                wandb_name=self.config.wandb_name,
                load_checkpoint_path=self.model_path if self._check_if_use_checkpoint() else None,
            )
            _reinforce_train(config)
            return self._get_checkpoint_path(log_path)
        finally:
            os.unlink(train_path)
            
    def _load_state(self) -> None:
        for model_name in TINKER_SUPPORTED_MODELS:
            self.models[model_name.split("/")[-1]] = model_name
        if os.path.exists(self.config.state_path):
            try:
                with open(self.config.state_path) as f:
                    state = json.load(f)
                self.models.update(state.get("models", {}))
                self.latest_update = state.get("latest_update")
                self.model_path = state.get("model_path")
                log.info(
                    "Loaded state from %s: %d model slugs, model_path=%s, latest_update=%s",
                    self.config.state_path,
                    len(self.models),
                    self.model_path,
                    "present" if self.latest_update is not None else "none",
                )
            except Exception:
                log.exception("Failed to load state from %s", self.config.state_path)
    
    def _save_state(self) -> None:
        Path(self.config.state_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            state = {
                "models": self.models,
                "latest_update": self.latest_update,
                "model_path": self.model_path,
            }
            with open(self.config.state_path, "w") as f:
                json.dump(state, f)
            log.info(
                "Saved state to %s: %d model slugs, model_path=%s, latest_update=%s",
                self.config.state_path,
                len(self.models),
                self.model_path,
                "present" if self.latest_update is not None else "none",
            )
        except Exception:
            log.exception("Failed to save state to %s", self.config.state_path)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    assert args.config.endswith(".yaml"), "Config file must be a YAML file"
    
    config = Config.from_yaml(args.config)
    server = Server(config)
    app = server._build_app()
    uvicorn.run(app, host=config.proxy_host, port=config.proxy_port)