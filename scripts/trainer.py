import tinker
import asyncio
import torch
import logging
import datetime

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.rl.train import (
    save_checkpoint_and_get_sampling_client,
)
from tinker_cookbook.supervised.train import run_evals
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import iteration_dir

try:
    from tinker_reinforce import (
        Config as REINFORCEConfig,
        make_reinforce_loss_fn,
        print_example as _REINFORCE_print_example,
        _save_baseline_state,
    )
    REINFORCE_IMPORT_ERROR: Exception | None = None
except Exception as e:  # noqa: BLE001
    REINFORCEConfig = Any
    make_reinforce_loss_fn = None
    _REINFORCE_print_example = None
    _save_baseline_state = None
    REINFORCE_IMPORT_ERROR = e

try:
    from tinker_dpo import (
        Config as DPOConfig,
        compute_dpo_loss,
        print_example as _DPO_print_example,
    )
    DPO_IMPORT_ERROR: Exception | None = None
except Exception as e:  # noqa: BLE001
    DPOConfig = Any
    compute_dpo_loss = None
    _DPO_print_example = None
    DPO_IMPORT_ERROR = e
from weight.train.run_opd import (
    Config as ArtifactOPDConfig,
    OnlineOPDRolloutDataset,
    _build_offline_topk_datums_async,
    _count_topk_supervision_tokens,
    _sample_online_artifact_datums_async,
)


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Checkpoint paths produced by a training round.

    ``sampler_path`` is servable by inference. ``state_path`` is resumable by
    ``create_training_client_from_state``.
    """

    sampler_path: str
    state_path: str


class Trainer():
    """Base class for long-lived online-RL trainers.

    A ``Trainer`` holds a live ``tinker.TrainingClient`` plus any auxiliary
    state (optimizer step counter, running baselines, loggers, etc.) across
    many training rounds. The server calls ``do_update(dataset)`` each time
    a new batch of sessions arrives; the trainer runs a small number of
    gradient steps and returns a fresh sampler-ready checkpoint path that
    the server can swap onto.

    Subclasses implement ``do_update`` and ``stop``. ``step`` is an internal
    helper whose signature is free to vary per mode (DPO wants epoch/batch
    indices; REINFORCE wants advantages; OPD wants a teacher client), so
    it's deliberately not part of the base contract.
    """

    async def do_update(self, dataset: SupervisedDataset) -> TrainingCheckpoint:
        raise NotImplementedError("Subclass must implement do_update")
    
    async def save_state(self):
        pass

    async def load_state(self):
        pass
    
    async def stop(self):
        raise NotImplementedError("Subclass must implement stop")


class REINFORCETrainer(Trainer):
    """Long-lived online REINFORCE trainer.

    Mirrors the structure of :class:`DPOTrainer` but with REINFORCE-specific
    state:

    * No reference sampling client -- variance reduction comes from a running
      baseline (the mean reward of the previous batch) rather than a KL term.
    * Per-trajectory rewards are read from the dataset via
      ``get_batch_rewards``; advantages are ``R_j - baseline``.
    * ``self.baseline`` persists across ``do_update`` calls so the first batch
      of a new round reuses the last round's average reward.
    """

    def __init__(
        self,
        logger: logging.Logger,
        config: REINFORCEConfig,
        training_client: tinker.TrainingClient,
        service_client: tinker.ServiceClient,
    ):
        if REINFORCE_IMPORT_ERROR is not None:
            raise RuntimeError("REINFORCE trainer dependencies failed to import") from REINFORCE_IMPORT_ERROR
        self.logger = logger
        self.total_steps = config.max_steps if config.max_steps is not None else 100_000  # TODO: fix the hardcoded value?
        self.config = config
        self.training_client = training_client
        self.service_client = service_client
        self.ml_logger = ml_log.setup_logging(
            log_dir=config.log_path,
            wandb_project=config.wandb_project,
            wandb_name=config.wandb_name,
            config=config,
            do_configure_logging_module=True,
        )
        self.log_path = config.log_path
        self.tokenizer = get_tokenizer(config.model_name)

        self.rolling_mgr = checkpoint_utils.RollingCheckpointManager(
            training_client=training_client,
            service_client=service_client,
            log_path=config.log_path,
            rolling_save_every=config.rolling_save_every,
            save_every=config.save_every,
            rolling_ttl_seconds=config.rolling_ttl_seconds,
        )

        # Global counters persisted across do_update calls (see DPOTrainer).
        self.step_idx = 0
        self.round_idx = 0

        # Running reward baseline. Initialized from config; updated to the
        # batch mean reward after every optimizer step so the next step sees
        # a freshly-shifted advantage.
        self.baseline = float(config.initial_baseline)

        if config.enable_trace:
            trace_events_path = str(Path(config.log_path) / "trace_events.jsonl")
            self.logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
            self.logger.info(
                f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
            )
            trace.trace_init(output_file=trace_events_path)

    async def do_update(
        self,
        dataset: SupervisedDataset,
    ) -> TrainingCheckpoint:
        """Run ``num_epochs`` passes over ``dataset`` and save a checkpoint.

        The dataset is expected to expose ``get_batch(i)`` (standard
        ``SupervisedDataset`` interface) as well as ``get_batch_rewards(i)``,
        which :class:`~tinker_formatter.ReinforceDataset` provides.

        Returns sampler and state paths for the final checkpoint produced this round.
        """
        self.round_idx += 1
        n_batches = len(dataset)
        round_start_step = self.step_idx

        reached_max_steps = False
        for epoch_idx in range(self.config.num_epochs):
            self.logger.info(
                "Round %d, epoch %d (step_idx=%d, n_batches=%d, baseline=%.4f)",
                self.round_idx, epoch_idx, self.step_idx, n_batches, self.baseline,
            )
            # Vary the shuffle seed across rounds/epochs (same convention as DPOTrainer).
            dataset.set_epoch(seed=self.round_idx * 1000 + epoch_idx)

            for batch_idx in range(n_batches):
                if (
                    self.config.max_steps is not None
                    and self.step_idx >= self.config.max_steps
                ):
                    reached_max_steps = True
                    break
                await self.step(epoch_idx=epoch_idx, batch_idx=batch_idx, dataset=dataset)
            if reached_max_steps:
                break

        # Always save a sampler-ready checkpoint at the end of the round (same
        # rationale as DPOTrainer: the server needs something concrete to swap onto).
        save_name = (
            f"round_{self.round_idx:06d}_reinforce_"
            f"{self.config.model_name.split('/')[-1]}_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        paths = await checkpoint_utils.save_checkpoint_async(
            training_client=self.training_client,
            name=save_name,
            log_path=self.config.log_path,
            kind="both",
            loop_state={"round": self.round_idx, "step": self.step_idx},
            ttl_seconds=None,
        )
        self.logger.info(
            "Round %d complete: %d step(s) taken this round, step_idx=%d, baseline=%.4f",
            self.round_idx, self.step_idx - round_start_step, self.step_idx, self.baseline,
        )

        sampler_path = paths.get("sampler_path")
        state_path = paths.get("state_path")
        if sampler_path is None or state_path is None:
            raise RuntimeError(
                f"Round {self.round_idx} produced incomplete checkpoint paths in {self.config.log_path}: {paths}"
            )
        return TrainingCheckpoint(sampler_path=sampler_path, state_path=state_path)

    async def step(
        self,
        epoch_idx: int,
        batch_idx: int,
        dataset: SupervisedDataset,
    ):
        """Perform a single REINFORCE training update step.

        Handles periodic checkpointing, baseline-state persistence, the
        forward-backward pass with the REINFORCE loss (``-A_j * seq_logprob_j``),
        the optimizer step, and metric logging for one batch. Also updates
        ``self.baseline`` to the mean reward of the processed batch.
        """
        step = self.step_idx
        metrics: dict[str, int | float | str] = {
            "round": self.round_idx,
            "epoch": epoch_idx,
        }

        with trace.trace_iteration(step=step) as window:
            # Mirror the offline script's behavior so baseline state on disk
            # tracks the same (epoch, batch) position checkpoint_utils records.
            _save_baseline_state(self.log_path, epoch_idx, batch_idx, self.baseline)

            # Save checkpoint if needed
            if self.config.save_every > 0 and step % self.config.save_every == 0 and step > 0:
                with trace.scope_span_sync("save_checkpoint"):
                    save_result = await checkpoint_utils.save_checkpoint_async(
                        training_client=self.training_client,
                        name=f"{step:06d}",
                        log_path=self.log_path,
                        kind="both",
                        loop_state={"epoch": epoch_idx, "batch": batch_idx},
                        ttl_seconds=self.config.ttl_seconds,
                    )
                if "state_path" in save_result:
                    metrics["state_path"] = save_result["state_path"]

            if self.rolling_mgr is not None:
                self.rolling_mgr.maybe_save(step=step, loop_state={"epoch": epoch_idx, "batch": batch_idx})

            learning_rate = self.config.learning_rate * compute_schedule_lr_multiplier(
                lr_schedule=self.config.lr_schedule, step=step, total_steps=self.total_steps
            )
            adam_params = tinker.AdamParams(
                learning_rate=learning_rate,
                beta1=self.config.adam_beta1,
                beta2=self.config.adam_beta2,
                eps=self.config.adam_eps,
            )

            # Prepare batch (data + per-trajectory rewards)
            with trace.scope_span_sync("get_batch"):
                data = dataset.get_batch(batch_idx)
                rewards = dataset.get_batch_rewards(batch_idx)

            # Print a few examples on the very first step of the trainer's life
            if step == 0:
                for i in range(min(3, len(data))):
                    _REINFORCE_print_example(
                        data[i], self.tokenizer, f"Example {i} (reward={rewards[i]:.3f})"
                    )

            # A_j = R_j - baseline, using the baseline from before this step
            advantages = [r - self.baseline for r in rewards]
            loss_fn = make_reinforce_loss_fn(advantages)

            async with trace.scope_span("step"):
                fb_future = await self.training_client.forward_backward_custom_async(data, loss_fn)
                backward_result = await fb_future.result_async()
                reinforce_metrics = backward_result.metrics
                optim_future = await self.training_client.optim_step_async(adam_params)
                await optim_future.result_async()

            # Update baseline to the mean reward of this batch for the next step
            new_baseline = sum(rewards) / len(rewards) if rewards else self.baseline
            self.baseline = new_baseline

            metrics.update(
                num_trajectories=len(data),
                num_tokens=sum(datum.model_input.length for datum in data),
                learning_rate=learning_rate,
                progress=step / self.total_steps,
                baseline=new_baseline,
                mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
                **reinforce_metrics,
            )

        # Log timing metrics from trace_iteration window
        metrics.update(window.get_timing_metrics())
        window.write_spans_jsonl(Path(self.log_path) / "timing_spans.jsonl", step=step)
        if self.config.span_chart_every > 0 and step % self.config.span_chart_every == 0:
            iter_dir = iteration_dir(self.log_path, step)
            if iter_dir is not None:
                iter_dir.mkdir(parents=True, exist_ok=True)
                trace.save_gantt_chart_html(window, step, iter_dir / "timing_gantt.html")
        self.ml_logger.log_metrics(metrics=metrics, step=step)

        self.step_idx += 1

    async def stop(self):
        self.rolling_mgr.finalize()
        self.ml_logger.close()
        self.logger.info("REINFORCE trainer terminated")


class DPOTrainer(Trainer):
    def __init__(
        self, 
        logger: logging.Logger,
        config: DPOConfig,
        training_client: tinker.TrainingClient,
        service_client: tinker.ServiceClient,
        reference_client: tinker.SamplingClient,
    ):
        if DPO_IMPORT_ERROR is not None:
            raise RuntimeError("DPO trainer dependencies failed to import") from DPO_IMPORT_ERROR
        self.logger = logger
        self.total_steps = config.max_steps if config.max_steps is not None else 100_000  # TODO: fix the hardcoded value?
        self.config = config
        self.training_client = training_client
        self.service_client = service_client
        self.reference_client = reference_client
        self.evaluators = [evaluator() for evaluator in config.evaluator_builders]
        self.infrequent_evaluators = [evaluator() for evaluator in config.infrequent_evaluator_builders]
        self.ml_logger = ml_log.setup_logging(
            log_dir=config.log_path,
            wandb_project=config.wandb_project,
            wandb_name=config.wandb_name,
            config=config,
            do_configure_logging_module=True,
        )
        self.log_path = config.log_path
        self.tokenizer = get_tokenizer(config.model_name)

        self.rolling_mgr = checkpoint_utils.RollingCheckpointManager(
            training_client=training_client,
            service_client=service_client,
            log_path=config.log_path,
            rolling_save_every=config.rolling_save_every,
            save_every=config.save_every,
            rolling_ttl_seconds=config.rolling_ttl_seconds,
        )
        
        # Global counters that persist across do_update calls. step_idx is
        # used for LR scheduling, checkpoint cadence, and W&B x-axis; round_idx
        # bumps once per do_update and is useful for naming final saves so
        # each training round produces a distinct checkpoint.
        self.step_idx = 0
        self.round_idx = 0

        # Initialize tracing
        if config.enable_trace:
            trace_events_path = str(Path(config.log_path) / "trace_events.jsonl")
            self.logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
            self.logger.info(
                f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
            )
            trace.trace_init(output_file=trace_events_path)
        
    async def do_update(
        self, 
        dataset: SupervisedDataset,
    ) -> TrainingCheckpoint:
        """Run ``num_epochs`` passes over the incoming ``dataset`` and save a checkpoint.

        In online RL this is called once per arriving batch of session data.
        The trainer's in-memory counters (``self.step_idx``, ``self.round_idx``)
        are the single source of truth for scheduling and naming; we do not
        read ``get_last_checkpoint`` here because the disk state can only
        ever be stale relative to the live ``TrainingClient``.

        Returns sampler and state paths for the checkpoint produced by this round.
        """
        self.round_idx += 1
        n_batches = len(dataset)
        round_start_step = self.step_idx

        reached_max_steps = False
        for epoch_idx in range(self.config.num_epochs):
            self.logger.info(
                "Round %d, epoch %d (step_idx=%d, n_batches=%d)",
                self.round_idx, epoch_idx, self.step_idx, n_batches,
            )
            # Vary the shuffle seed across rounds and epochs so repeated
            # passes over a tiny batch aren't perfectly correlated.
            dataset.set_epoch(seed=self.round_idx * 1000 + epoch_idx)

            for batch_idx in range(n_batches):
                if (
                    self.config.max_steps is not None
                    and self.step_idx >= self.config.max_steps
                ):
                    reached_max_steps = True
                    break
                await self.step(epoch_idx=epoch_idx, batch_idx=batch_idx, dataset=dataset)
            if reached_max_steps:
                break

        # Always save a sampler-ready checkpoint at the end of the round so
        # the server has something concrete to swap onto, even if max_steps
        # was reached and the inner loop didn't execute a step this round.
        save_name = (
            f"round_{self.round_idx:06d}_dpo_"
            f"{self.config.model_name.split('/')[-1]}_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        paths = await checkpoint_utils.save_checkpoint_async(
            training_client=self.training_client,
            name=save_name,
            log_path=self.config.log_path,
            kind="both",
            loop_state={"round": self.round_idx, "step": self.step_idx},
            ttl_seconds=None,
        )
        self.logger.info(
            "Round %d complete: %d step(s) taken this round, step_idx=%d",
            self.round_idx, self.step_idx - round_start_step, self.step_idx,
        )

        sampler_path = paths.get("sampler_path")
        state_path = paths.get("state_path")
        if sampler_path is None or state_path is None:
            raise RuntimeError(
                f"Round {self.round_idx} produced incomplete checkpoint paths in {self.config.log_path}: {paths}"
            )
        return TrainingCheckpoint(sampler_path=sampler_path, state_path=state_path)
        
    async def step(
        self,
        epoch_idx: int,
        batch_idx: int,
        dataset: SupervisedDataset,
    ):
        """Perform a single DPO training update step.

        Handles periodic checkpointing, evaluation, reference log-prob
        computation, the forward-backward pass with the custom DPO loss,
        the optimizer step, and metric logging for one batch.

        Args:
            epoch_idx (int): Epoch index within the current ``do_update``
                round (zero-based). For online RL this is almost always 0.
            batch_idx (int): Batch index within the epoch.
            dataset (SupervisedDataset): Training dataset providing batches
                of interleaved chosen/rejected ``Datum`` pairs.
        """
        step = self.step_idx
        metrics: dict[str, int | float | str] = {
            "round": self.round_idx,
            "epoch": epoch_idx,
        }

        with trace.trace_iteration(step=step) as window:
            # Save checkpoint if needed
            if self.config.save_every > 0 and step % self.config.save_every == 0 and step > 0:
                with trace.scope_span_sync("save_checkpoint"):
                    save_result = await checkpoint_utils.save_checkpoint_async(
                        training_client=self.training_client,
                        name=f"{step:06d}",
                        log_path=self.log_path,
                        kind="both",
                        loop_state={"epoch": epoch_idx, "batch": batch_idx},
                        ttl_seconds=self.config.ttl_seconds,
                    )
                if "state_path" in save_result:
                    metrics["state_path"] = save_result["state_path"]

            if self.rolling_mgr is not None:
                self.rolling_mgr.maybe_save(step=step, loop_state={"epoch": epoch_idx, "batch": batch_idx})

            learning_rate = self.config.learning_rate * compute_schedule_lr_multiplier(
                lr_schedule=self.config.lr_schedule, step=step, total_steps=self.total_steps
            )
            adam_params = tinker.AdamParams(
                learning_rate=learning_rate,
                beta1=self.config.adam_beta1,
                beta2=self.config.adam_beta2,
                eps=self.config.adam_eps,
            )

            # Evaluation
            if self.config.eval_every > 0 and step % self.config.eval_every == 0:
                with trace.scope_span_sync("evals"):
                    eval_metrics = await run_evals(self.evaluators, self.training_client, step)
                metrics.update(eval_metrics)

            if self.config.infrequent_eval_every > 0 and step % self.config.infrequent_eval_every == 0:
                with trace.scope_span_sync("infrequent_evals"):
                    eval_metrics = await run_evals(self.infrequent_evaluators, self.training_client, step)
                metrics.update(eval_metrics)

            # Prepare batch
            with trace.scope_span_sync("get_batch"):
                data = dataset.get_batch(batch_idx)

            # Split data into chosen and rejected pairs
            chosen_data = [datum for i, datum in enumerate(data) if i % 2 == 0]
            rejected_data = [datum for i, datum in enumerate(data) if i % 2 == 1]

            # Print example for first batch
            if step == 0:
                for i in range(min(10, len(chosen_data))):
                    _DPO_print_example(chosen_data[i], self.tokenizer, "Chosen")
                    _DPO_print_example(rejected_data[i], self.tokenizer, "Rejected")

            with trace.scope_span_sync("get_ref_logprobs"):
                # Get reference log probabilities
                # Need to reconstruct full sequences for the sampling client
                full_sequences = []
                for datum in data:
                    # Reconstruct the full sequence by appending the last target token
                    target_tokens = datum.loss_fn_inputs["target_tokens"].data
                    if target_tokens:
                        full_sequence = datum.model_input.append_int(int(target_tokens[-1]))
                        full_sequences.append(full_sequence)
                    else:
                        # If no target tokens, just use the model input as is
                        full_sequences.append(datum.model_input)

                # Compute reference log probabilities in parallel.
                all_ref_logprobs = await asyncio.gather(
                    *[self.reference_client.compute_logprobs_async(seq) for seq in full_sequences]
                )

                # Extract the relevant logprobs (skip the first token which is the prompt)
                all_ref_logprob_seqs = [torch.tensor(logprobs[1:]) for logprobs in all_ref_logprobs]

                # Split reference results into chosen and rejected
                chosen_ref_logprob_seqs = [all_ref_logprob_seqs[i] for i in range(0, len(data), 2)]
                rejected_ref_logprob_seqs = [all_ref_logprob_seqs[i] for i in range(1, len(data), 2)]

            # Create DPO loss function
            def dpo_loss_fn(
                data: list[tinker.Datum], logprobs_list: list[torch.Tensor]
            ) -> tuple[torch.Tensor, dict[str, float]]:
                # Split logprobs into chosen and rejected
                chosen_logprob_seqs = [logprobs_list[i] for i in range(0, len(data), 2)]
                rejected_logprob_seqs = [logprobs_list[i] for i in range(1, len(data), 2)]

                # Extract log probabilities
                chosen_logprobs = []
                chosen_ref_logprobs = []
                rejected_logprobs = []
                rejected_ref_logprobs = []

                for i in range(len(chosen_data)):
                    # Compute weighted logprobs for chosen responses
                    chosen_logprob_seq = chosen_logprob_seqs[i]
                    chosen_ref_logprob_seq = chosen_ref_logprob_seqs[i]
                    chosen_weights = torch.tensor(chosen_data[i].loss_fn_inputs["weights"].data)
                    chosen_logprob = torch.dot(chosen_logprob_seq.float(), chosen_weights.float())
                    chosen_ref_logprob = torch.dot(
                        chosen_ref_logprob_seq.float(), chosen_weights.float()
                    )
                    chosen_logprobs.append(chosen_logprob)
                    chosen_ref_logprobs.append(chosen_ref_logprob)

                    # Compute weighted logprobs for rejected responses
                    rejected_logprob_seq = rejected_logprob_seqs[i]
                    rejected_ref_logprob_seq = rejected_ref_logprob_seqs[i]
                    rejected_weights = torch.tensor(rejected_data[i].loss_fn_inputs["weights"].data)
                    rejected_logprob = torch.dot(rejected_logprob_seq.float(), rejected_weights.float())
                    rejected_ref_logprob = torch.dot(
                        rejected_ref_logprob_seq.float(), rejected_weights.float()
                    )
                    rejected_logprobs.append(rejected_logprob)
                    rejected_ref_logprobs.append(rejected_ref_logprob)

                # Compute DPO loss
                return compute_dpo_loss(
                    chosen_logprobs=chosen_logprobs,
                    rejected_logprobs=rejected_logprobs,
                    chosen_ref_logprobs=chosen_ref_logprobs,
                    rejected_ref_logprobs=rejected_ref_logprobs,
                    dpo_beta=self.config.dpo_beta,
                )

            async with trace.scope_span("step"):
                fb_future = await self.training_client.forward_backward_custom_async(data, dpo_loss_fn)
                backward_result = await fb_future.result_async()
                dpo_metrics = backward_result.metrics

                optim_future = await self.training_client.optim_step_async(adam_params)
                await optim_future.result_async()

            # Prepare metrics
            metrics.update(
                num_pairs=len(chosen_data),
                num_tokens=sum(datum.model_input.length for datum in data),
                learning_rate=learning_rate,
                progress=step / self.total_steps,
                **dpo_metrics,
            )

        # Log timing metrics from trace_iteration window
        metrics.update(window.get_timing_metrics())
        window.write_spans_jsonl(Path(self.log_path) / "timing_spans.jsonl", step=step)
        if self.config.span_chart_every > 0 and step % self.config.span_chart_every == 0:
            iter_dir = iteration_dir(self.log_path, step)
            if iter_dir is not None:
                iter_dir.mkdir(parents=True, exist_ok=True)
                trace.save_gantt_chart_html(window, step, iter_dir / "timing_gantt.html")
        self.ml_logger.log_metrics(metrics=metrics, step=step)
        
        # Increment step index
        self.step_idx += 1
        
    async def stop(self):
        self.rolling_mgr.finalize()
        self.ml_logger.close()
        self.logger.info("DPO trainer terminated")
                

class OPDTrainer(Trainer):
    """Long-lived online artifact OPD trainer for the server.

    This keeps Aspen's server-friendly packaging (persistent training client,
    teacher client, student sampler refresh, step/round counters, final
    checkpoint handoff), but replaces the prompt-only SDFT data path with the
    weight-format artifact OPD path from ``weight.train.run_opd``.
    """

    def __init__(
        self,
        logger: logging.Logger,
        config: ArtifactOPDConfig,
        training_client: tinker.TrainingClient,
        service_client: tinker.ServiceClient,
        max_context_length: int = 32768,
    ):
        self.logger = logger
        self.total_steps = config.max_steps if config.max_steps is not None else 100_000  # TODO: fix the hardcoded value?
        self.config = config
        self.training_client = training_client
        self.service_client = service_client
        self.max_context_length = max_context_length
        self.ml_logger = ml_log.setup_logging(
            log_dir=config.log_path,
            wandb_project=config.wandb_project,
            wandb_name=config.wandb_name,
            config=config,
        )
        self.log_path = config.log_path
        self.tokenizer = get_tokenizer(config.model_name)

        if config.topk <= 0:
            raise ValueError("Server OPD currently supports artifact top-K mode only; set opd_topk > 0")
        assert config.renderer_name is not None, "OPDTrainer requires config.renderer_name"
        self.renderer = renderers.get_renderer(config.renderer_name, tokenizer=self.tokenizer)

        # Static frozen teacher, matching the weight OPD script.
        self.teacher_client: tinker.SamplingClient = service_client.create_sampling_client(
            base_model=config.model_name
        )
        self.logger.info(f"Created static teacher sampling client for {config.model_name}")

        # Student sampling client is created lazily on the first do_update
        # because save_checkpoint_and_get_sampling_client is async.
        self.sampling_client: tinker.SamplingClient | None = None

        # Global counters persisted across do_update calls (see DPOTrainer).
        self.step_idx = 0
        self.round_idx = 0

        if config.enable_trace:
            trace_events_path = str(Path(config.log_path) / "trace_events.jsonl")
            self.logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
            self.logger.info(
                f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
            )
            trace.trace_init(output_file=trace_events_path)

    async def _ensure_sampling_client(self) -> None:
        """Lazily create the initial student sampling client.

        Done here rather than in ``__init__`` because
        ``save_checkpoint_and_get_sampling_client`` is async.
        """
        if self.sampling_client is None:
            self.sampling_client, _ = await save_checkpoint_and_get_sampling_client(
                self.training_client,
                self.step_idx,
                self.config.log_path,
                self.config.save_every,
            )

    async def do_update(
        self,
        dataset: OnlineOPDRolloutDataset,
        num_epochs: int | None = None,
    ) -> TrainingCheckpoint:
        """Run online artifact OPD over the freshly queued session batch."""
        await self._ensure_sampling_client()

        self.round_idx += 1
        n_batches = len(dataset)
        epochs = num_epochs if num_epochs is not None else self.config.num_epochs
        round_start_step = self.step_idx

        self.logger.info(
            "Round %d: step_idx=%d, n_batches=%d, epochs=%d",
            self.round_idx, self.step_idx, n_batches, epochs,
        )

        for epoch_idx in range(epochs):
            dataset.set_epoch(seed=self.round_idx * 1000 + epoch_idx)
            for batch_idx in range(n_batches):
                if (
                    self.config.max_steps is not None
                    and self.step_idx >= self.config.max_steps
                ):
                    break
                await self.step(
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                    dataset=dataset,
                    total_steps=max(1, self.total_steps),
                )

        # Final sampler-ready checkpoint for the server to swap onto. Same
        # rationale as DPOTrainer: always produce one, even if max_steps
        # caused the inner loop to be a no-op this round.
        save_name = (
            f"round_{self.round_idx:06d}_opd_"
            f"{self.config.model_name.split('/')[-1]}_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        paths = await checkpoint_utils.save_checkpoint_async(
            training_client=self.training_client,
            name=save_name,
            log_path=self.config.log_path,
            kind="both",
            loop_state={"round": self.round_idx, "step": self.step_idx},
            ttl_seconds=None,
        )
        self.logger.info(
            "Round %d complete: %d step(s) taken this round, step_idx=%d",
            self.round_idx, self.step_idx - round_start_step, self.step_idx,
        )

        sampler_path = paths.get("sampler_path")
        state_path = paths.get("state_path")
        if sampler_path is None or state_path is None:
            raise RuntimeError(
                f"Round {self.round_idx} produced incomplete checkpoint paths in {self.config.log_path}: {paths}"
            )
        return TrainingCheckpoint(sampler_path=sampler_path, state_path=state_path)

    async def step(
        self,
        epoch_idx: int,
        batch_idx: int,
        dataset: OnlineOPDRolloutDataset,
        total_steps: int,
    ) -> None:
        """Sample artifact completions, filter them, then train top-K OPD."""
        assert self.sampling_client is not None, (
            "step() invoked before _ensure_sampling_client(); call do_update() as the entry point"
        )

        step = self.step_idx
        metrics: dict[str, Any] = {
            "round": self.round_idx,
            "epoch": epoch_idx,
            "progress/batch": batch_idx,
        }

        with trace.trace_iteration(step=step) as window:
            async with trace.scope_span("sample"):
                rows = dataset.get_batch(batch_idx)
                student_datums, teacher_prompt_inputs, rollout_metrics = await _sample_online_artifact_datums_async(
                    rows,
                    self.renderer,
                    self.sampling_client,
                    max_tokens=self.config.rollout_max_tokens,
                    temperature=self.config.rollout_temperature,
                    attempts=self.config.rollout_attempts,
                    max_length=dataset._max_length,
                    step=step,
                    sample_log_path=(
                        Path(self.config.log_path) / "online_rollout_samples.jsonl"
                        if self.config.log_rollout_samples else None
                    ),
                    sample_log_chars=self.config.rollout_sample_log_chars,
                )
            metrics.update(rollout_metrics)

            if not student_datums:
                learning_rate = self.config.learning_rate * compute_schedule_lr_multiplier(
                    lr_schedule=self.config.lr_schedule,
                    step=step,
                    total_steps=total_steps,
                )
                metrics.update(
                    {
                        "opd_online/no_valid_batch": 1.0,
                        "learning_rate": learning_rate,
                        "progress": step / max(total_steps, 1),
                    }
                )
                self.ml_logger.log_metrics(metrics=metrics, step=step)
                self.logger.warning(
                    "Skipping OPD step %d: no valid artifact samples (filter_rate=%.3f)",
                    step,
                    rollout_metrics.get("opd_online/filter_rate", 0.0),
                )
                self.step_idx += 1
                return

            async with trace.scope_span("build_topk_distillation_datums"):
                topk_datums, topk_metrics = await _build_offline_topk_datums_async(
                    student_datums,
                    teacher_prompt_inputs,
                    self.teacher_client,
                    topk=self.config.topk,
                    max_context_length=self.max_context_length,
                    vocab_size=len(self.tokenizer),
                    teacher_temperature=self.config.teacher_temperature,
                )
            metrics.update(topk_metrics)

            if self.config.save_every > 0 and step % self.config.save_every == 0 and step > 0:
                save_result = await checkpoint_utils.save_checkpoint_async(
                    training_client=self.training_client,
                    name=f"{step:06d}",
                    log_path=self.log_path,
                    kind="both",
                    loop_state={"epoch": epoch_idx, "batch": batch_idx},
                    ttl_seconds=self.config.ttl_seconds,
                )
                if "state_path" in save_result:
                    metrics["state_path"] = save_result["state_path"]

            learning_rate = self.config.learning_rate * compute_schedule_lr_multiplier(
                lr_schedule=self.config.lr_schedule,
                step=step,
                total_steps=total_steps,
            )
            adam_params = tinker.AdamParams(
                learning_rate=learning_rate,
                beta1=self.config.adam_beta1,
                beta2=self.config.adam_beta2,
                eps=self.config.adam_eps,
            )

            async with trace.scope_span("train"):
                fb_future = await self.training_client.forward_backward_async(
                    topk_datums,
                    loss_fn="cross_entropy",
                )
                backward_result = await fb_future.result_async()
                optim_future = await self.training_client.optim_step_async(adam_params)
                await optim_future.result_async()

            result_metrics = backward_result.metrics
            if not isinstance(result_metrics, dict):
                try:
                    result_metrics = dict(result_metrics.items())
                except Exception:
                    result_metrics = {}

            loss_sum = float(result_metrics.get("loss:sum", result_metrics.get("loss", 0.0)))
            batch_tokens = _count_topk_supervision_tokens(topk_datums, topk=self.config.topk)
            per_token_ce = loss_sum / float(batch_tokens)
            metrics.update(
                {
                    "opd_loss": per_token_ce,
                    "opd/loss_sum": loss_sum,
                    "opd/batch_completion_tokens": float(batch_tokens),
                    "opd/per_token_ce": per_token_ce,
                    "num_examples": len(topk_datums),
                    "learning_rate": learning_rate,
                    "progress": step / max(total_steps, 1),
                    **result_metrics,
                }
            )

            # Refresh the student sampling client onto the just-updated weights.
            self.sampling_client, sampler_metrics = await save_checkpoint_and_get_sampling_client(
                self.training_client,
                step + 1,
                self.config.log_path,
                self.config.save_every,
            )
            metrics.update(sampler_metrics)
            self.logger.info(
                "Online OPD step %d: valid=%d/%d filter_rate=%.3f loss=%.4f",
                step,
                int(rollout_metrics.get("opd_online/valid_examples", 0.0)),
                int(rollout_metrics.get("opd_online/batch_examples", 0.0)),
                rollout_metrics.get("opd_online/filter_rate", 0.0),
                per_token_ce,
            )

        # Log timing metrics from trace_iteration window.
        metrics.update(window.get_timing_metrics())
        window.write_spans_jsonl(Path(self.log_path) / "timing_spans.jsonl", step=step)
        if self.config.span_chart_every > 0 and step % self.config.span_chart_every == 0:
            trace.save_gantt_chart_html(
                window, step, Path(self.log_path) / f"timing_gantt_{step:06d}.html"
            )
        self.ml_logger.log_metrics(metrics, step=step)

        self.step_idx += 1

    async def stop(self):
        self.ml_logger.close()
        self.logger.info("OPD trainer terminated")
