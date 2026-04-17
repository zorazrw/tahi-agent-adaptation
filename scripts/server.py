import json
import os
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
from fastapi.responses import JSONResponse, StreamingResponse
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


@dataclass
class Config:
    # -- Mode --
    mode: Literal["dpo", "opd", "reinforce"] = "reinforce"
    
    # -- Model --
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str = "qwen3_instruct"
    lora_rank: int = 32
    
    # -- Inference --
    temperature: float = 0.0
    logprobs: bool = False
    
    # -- Update --
    update_every_n_sessions: int = 1 # update the model after n new sessions are processed
    
    # -- Training --
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
        self.model_path = config.model_name
        self.sessions_queue = asyncio.Queue()
        self.training_queue = asyncio.Queue()
        self.training_event = asyncio.Event()
        self.training_lock = asyncio.Lock()
        
        # event to signal that the model is ready to be used
        self.model_ready = asyncio.Event()
        self.model_ready.set()
    
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
                return HTTPException(status_code=400, content={"error": "trajectory must be a non-empty list"})

            
            await self.sessions_queue.put({
                "uuid": session_id,
                "name": name,
                "trajectory": trajectory,
            })
            
            log.info("Session enqueued: id=%s name=%r steps=%d queue_depth=%d",
                     session_id, name, len(trajectory), self.sessions_queue.qsize())
            return {"ok": True, "session_id": session_id}
        
        @app.post("/v1/chat/completions")
        async def _handle_chat_completions(request: Request):
            # block until the model is ready to be used
            await self.model_ready.wait()
            
            body = await request.json()
            
            messages = body.pop("messages", [])
            stream = body.pop("stream", True)
            _ = body.pop("model", None)
            log.info("Chat completion request: model=%s stream=%s messages=%d",
                     self.model_path, stream, len(messages))
            
            # Override inference parameters
            body["temperature"] = self.config.temperature
            body["logprobs"] = self.config.logprobs
            
            if stream:
                rewrite = func_mapping.get(config.renderer_name)
                
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
                        await asyncio.wait_for(self.update_model_path(checkpoint, timeout=600.0), timeout=600.0)
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
        train_path = self._write_sessions_file(items)
        try:
            log_path = f"logs/tinker_dpo/{int(time.time())}"
            dataset_builder = DPODataBuilder(
                train_path=train_path,
                common_config=ChatDatasetBuilderCommonConfig(
                    model_name_for_tokenizer=self.config.model_name,
                    renderer_name=self.config.renderer_name,
                    max_length=None,
                    batch_size=self.config.batch_size,
                ),
            )
            config = DPOConfig(
                log_path=log_path,
                model_name=self.config.model_name,
                dataset_builder=dataset_builder,
                renderer_name=self.config.renderer_name,
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
        try:
            log_path = f"logs/tinker_opd/{int(time.time())}"
            tokenizer = get_tokenizer(self.config.model_name)
            renderer = renderers.get_renderer(self.config.renderer_name, tokenizer=tokenizer)
            sdft_dataset = OPDSDFTDataset.from_json(
                data_path=train_path,
                renderer=renderer,
                batch_size=self.config.batch_size,
            )
            config = OPDConfig(
                model_name=self.config.model_name,
                renderer_name=self.config.renderer_name,
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
        try:
            log_path = f"logs/tinker_reinforce/{int(time.time())}"
            dataset_builder = ReinforceDataBuilder(
                train_path=train_path,
                reward_alpha=self.config.reward_alpha,
                common_config=ChatDatasetBuilderCommonConfig(
                    model_name_for_tokenizer=self.config.model_name,
                    renderer_name=self.config.renderer_name,
                    max_length=None,
                    batch_size=self.config.batch_size,
                ),
            )
            config = ReinforceConfig(
                log_path=log_path,
                model_name=self.config.model_name,
                dataset_builder=dataset_builder,
                renderer_name=self.config.renderer_name,
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
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    assert args.config.endswith(".yaml"), "Config file must be a YAML file"
    
    config = Config.from_yaml(args.config)
    server = Server(config)
    app = server._build_app()
    uvicorn.run(app, host=config.proxy_host, port=config.proxy_port)