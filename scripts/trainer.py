import tinker
import asyncio
import torch
import logging
import datetime

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.rollouts import do_group_rollout_and_filter_constant_reward
from tinker_cookbook.rl.train import (
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.supervised.train import run_evals
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import iteration_dir

from tinker_reinforce import (
    Config as REINFORCEConfig,
    make_reinforce_loss_fn,
    print_example as _REINFORCE_print_example,
    _save_baseline_state,
)
from tinker_dpo import (
    Config as DPOConfig,
    compute_dpo_loss,
    print_example as _DPO_print_example,
)
from tinker_opd import (
    Config as OPDConfig,
    SDFTBatchProvider,
    build_sdft_teacher_prompt,
    build_topk_distillation_datums,
    compute_sdft_advantages,
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
    """Long-lived online OPD / SDFT trainer.

    Mirrors the shape of :class:`DPOTrainer` and :class:`REINFORCETrainer`
    but wraps the self-distillation loop from ``tinker_opd.main``:

    * The student generates on-policy rollouts; a (static or periodically
      synced) teacher sees the question *and* the golden answer and provides
      either top-K soft targets (``cfg.topk > 0``) or per-token importance
      weights (``cfg.topk == 0``).
    * Training uses ``tinker_cookbook.rl.train.train_step`` rather than a
      bespoke ``forward_backward_custom`` call, and the student
      ``SamplingClient`` is refreshed after every optimizer step via
      ``save_checkpoint_and_get_sampling_client``.
    * There is no reference client, no running baseline, no rolling
      checkpoint manager, and no LR schedule -- ``OPDConfig`` omits those
      knobs deliberately.

    Unlike the other trainers, ``dataset`` here is an :class:`SDFTBatchProvider`
    whose ``get_batch`` returns ``(builders, questions, golden_answers)``.
    """

    def __init__(
        self,
        logger: logging.Logger,
        config: OPDConfig,
        training_client: tinker.TrainingClient,
        service_client: tinker.ServiceClient,
    ):
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
        )
        self.log_path = config.log_path
        self.tokenizer = get_tokenizer(config.model_name)

        assert config.renderer_name is not None, (
            "OPDTrainer requires config.renderer_name (resolve before constructing the trainer)"
        )
        self.renderer = renderers.get_renderer(config.renderer_name, tokenizer=self.tokenizer)
        # Reasoning-aware renderers (Qwen3, Kimi K2, DeepSeek V3 thinking, ...)
        # carry a per-instance ``strip_thinking_from_history`` flag controlling
        # whether ``<think>...</think>`` survives in non-last assistant
        # messages. Surface OPDConfig.strip_thinking_from_history here so the
        # SDFT teacher can attend to the golden answer's chain-of-thought.
        if hasattr(self.renderer, "strip_thinking_from_history"):
            self.renderer.strip_thinking_from_history = config.strip_thinking_from_history
            self.logger.info(
                "Renderer %s: strip_thinking_from_history=%s",
                type(self.renderer).__name__,
                config.strip_thinking_from_history,
            )

        # Evaluators run every ``eval_every`` steps against the student
        # sampling client (same semantics as tinker_opd.main).
        self.evaluators: list[SamplingClientEvaluator] = [e() for e in config.evaluator_builders]

        # Static teacher sampling client. May be re-pointed at a fresh
        # snapshot of the student every ``teacher_sync_every`` steps.
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
        dataset: SDFTBatchProvider,
        num_epochs: int,
    ) -> TrainingCheckpoint:
        """Run one pass over the incoming batches and save a checkpoint.

        OPD's ``Config`` has no ``num_epochs`` -- each batch is consumed
        exactly once per round (matching ``tinker_opd.main``). The per-round
        sampler checkpoint follows the same naming conventions as DPO and
        REINFORCE. The paired state checkpoint lets the server resume training.
        """
        await self._ensure_sampling_client()

        self.round_idx += 1
        n_batches = len(dataset)
        round_start_step = self.step_idx

        self.logger.info(
            "Round %d: step_idx=%d, n_batches=%d",
            self.round_idx, self.step_idx, n_batches,
        )
        
        self.logger.info(f"Training for {num_epochs} epochs")
        for epoch_idx in range(num_epochs):
            for batch_idx in range(n_batches):
                if (
                    self.config.max_steps is not None
                    and self.step_idx >= self.config.max_steps
                ):
                    break
                await self.step(epoch_idx=epoch_idx, batch_idx=batch_idx, dataset=dataset)

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
        dataset: SDFTBatchProvider,
    ) -> None:
        """Perform a single OPD / SDFT training update step.

        Handles evaluation, on-policy rollouts with the current student
        sampling client, teacher-conditioned distillation target construction
        (top-K CE or importance-sampling advantages), the optimizer step via
        ``train_step``, refreshing the student sampling client, optional
        teacher hard-sync, and metric logging for one batch.

        ``epoch_idx`` is carried for signature parity with the other
        trainers; OPD always passes 0 because ``OPDConfig`` has no
        ``num_epochs`` knob.
        """
        assert self.sampling_client is not None, (
            "step() invoked before _ensure_sampling_client(); call do_update() as the entry point"
        )

        step = self.step_idx
        metrics: dict[str, Any] = {
            "round": self.round_idx,
            "epoch": epoch_idx,
            "progress/batch": batch_idx,
            "optim/lr": self.config.learning_rate,
        }

        with trace.trace_iteration(step=step) as window:
            # Evaluation against the *current* student sampling client
            if self.config.eval_every > 0 and step % self.config.eval_every == 0:
                async with trace.scope_span("run_evals"):
                    for evaluator in self.evaluators:
                        eval_metrics = await evaluator(self.sampling_client)
                        metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

            # Get batch: builders + questions + golden answers
            builders_P, questions_P, golden_answers_P = dataset.get_batch(batch_idx)

            # On-policy rollouts. Uses do_group_rollout so group_size > 1 and
            # multi-turn envs work without extra code; with group_size=1 this
            # collapses to a single sample_async per problem.
            async with trace.scope_span("sample"):
                trajectory_groups_raw = await asyncio.gather(
                    *[
                        asyncio.create_task(
                            do_group_rollout_and_filter_constant_reward(
                                self.sampling_client,
                                builder,
                                temperature=self.config.temperature,
                                max_tokens=self.config.max_tokens,
                                do_remove_constant_reward_groups=False,
                            ),
                            name=f"sample_task_{i}",
                        )
                        for i, builder in enumerate(builders_P)
                    ],
                )
            trajectory_groups_P: list[TrajectoryGroup] = [
                tg for tg in trajectory_groups_raw if tg is not None
            ]

            taglist_P = [b.logging_tags() for b in builders_P]
            metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

            # Advantages start as 0 here (rewards are all 0 for pure SDFT);
            # they'll be overwritten by the teacher-based target builder below.
            async with trace.scope_span("assemble_training_data"):
                advantages_P = compute_advantages(trajectory_groups_P)
                data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

            # Teacher prompts: one per problem, conditioned on the golden answer.
            teacher_prompts_P = [
                build_sdft_teacher_prompt(
                    question=question,
                    golden_answer=golden_answer,
                    renderer=self.renderer,
                    system_prompt=self.config.system_prompt,
                    demo_template=self.config.demo_template,
                    chat_redo_message=self.config.chat_redo_message,
                )
                for question, golden_answer in zip(questions_P, golden_answers_P)
            ]
            
            # Log rollouts and teacher prompts 
            for idx, datum in enumerate(data_D):
                self.logger.info(f"Example {idx}: ")
                self.logger.info("Student rollout: ")
                self.logger.info(colorize_example(datum, self.tokenizer, key="mask"))
                self.logger.info("Teacher prompt: ")
                teacher_prompt = self.renderer.tokenizer.decode(teacher_prompts_P[idx].to_ints())
                self.logger.info(teacher_prompt)

            # Optional: free-form teacher rollout for debugging. The teacher
            # samples from ``teacher_prompts_P`` (which already includes the
            # golden answer + redo) so we can eyeball whether the teacher's
            # "ideal" generation actually matches what we want the student to
            # learn. Gated behind a config flag + cadence to bound extra cost.
            do_teacher_rollout = (
                getattr(self.config, "debug_teacher_rollout", False)
                and self.teacher_client is not None
                and len(teacher_prompts_P) > 0
                and (
                    getattr(self.config, "debug_teacher_rollout_every", 0) <= 0
                    or step % self.config.debug_teacher_rollout_every == 0
                )
            )
            if do_teacher_rollout:
                async with trace.scope_span("debug_teacher_sample"):
                    teacher_sampling_params = tinker.SamplingParams(
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        stop=self.renderer.get_stop_sequences(),
                    )
                    teacher_samples_P = await asyncio.gather(
                        *[
                            self.teacher_client.sample_async(
                                prompt=tp,
                                num_samples=1,
                                sampling_params=teacher_sampling_params,
                            )
                            for tp in teacher_prompts_P
                        ]
                    )
                for idx, resp in enumerate(teacher_samples_P):
                    sequences = getattr(resp, "sequences", None) or []
                    if not sequences:
                        self.logger.info(f"Teacher rollout (Example {idx}): <empty>")
                        continue
                    out_tokens = list(getattr(sequences[0], "tokens", []) or [])
                    stop_reason = getattr(sequences[0], "stop_reason", None)
                    out_text = self.renderer.tokenizer.decode(out_tokens)
                    self.logger.info(
                        f"Teacher rollout (Example {idx}, "
                        f"n_tokens={len(out_tokens)}, stop={stop_reason}): "
                    )
                    self.logger.info("Teacher rollout: ")
                    self.logger.info(out_text)

            if self.config.topk > 0:
                # Top-K CE distillation (the validated path).
                async with trace.scope_span("build_topk_distillation_datums"):
                    topk_datums, topk_metrics = await build_topk_distillation_datums(
                        data_D,
                        metadata_D,
                        self.teacher_client,
                        teacher_prompts_P,
                        student_client=self.sampling_client,
                        topk=self.config.topk,
                        max_context_length=self.config.max_context_length,
                        vocab_size=len(self.tokenizer),
                    )
                metrics.update(topk_metrics)

                async with trace.scope_span("train"):
                    training_logprobs_D = await train_step(
                        data_D=topk_datums,
                        training_client=self.training_client,
                        learning_rate=self.config.learning_rate,
                        num_substeps=self.config.num_substeps,
                        loss_fn="cross_entropy",
                        metrics=metrics,
                    )
                    
                # Compute loss and log it
                
            else:
                # Importance-sampling fallback (topk=0): compute per-token
                # teacher_lp - student_lp advantages, then train with the
                # configured loss.
                async with trace.scope_span("compute_sdft_advantages"):
                    is_metrics = await compute_sdft_advantages(
                        data_D,
                        metadata_D,
                        self.teacher_client,
                        teacher_prompts_P,
                        max_context_length=self.config.max_context_length,
                    )
                metrics.update(is_metrics)

                async with trace.scope_span("train"):
                    await train_step(
                        data_D=data_D,
                        training_client=self.training_client,
                        learning_rate=self.config.learning_rate,
                        num_substeps=self.config.num_substeps,
                        loss_fn=self.config.loss_fn,
                        metrics=metrics,
                    )

            # Refresh the student sampling client onto the just-updated weights.
            # save_checkpoint_and_get_sampling_client handles save_every internally.
            self.sampling_client, _ = await save_checkpoint_and_get_sampling_client(
                self.training_client, step + 1, self.config.log_path, self.config.save_every
            )

            # Optional teacher hard-sync (approximates EMA at a coarse cadence).
            if self.config.teacher_sync_every and (step + 1) % self.config.teacher_sync_every == 0:
                sync_name = f"teacher_sync_{step + 1}"
                sync_future = await self.training_client.save_weights_for_sampler_async(sync_name)
                sync_result = await sync_future.result_async()
                self.teacher_client = self.service_client.create_sampling_client(
                    base_model=self.config.model_name, model_path=sync_result.path
                )
                self.logger.info(f"Synced teacher weights at step {step + 1}")

            metrics.update(
                num_trajectory_groups=len(trajectory_groups_P),
                progress=step / self.total_steps,
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