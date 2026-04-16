"""DPO training on weight-format session JSON.

Creates a ``WeightDPODataBuilder`` and delegates to ``tinker_dpo.main()``.

Usage::

    python -m scripts.weight.train.run_dpo \\
        --train-path data/weight.json \\
        --model-name Qwen/Qwen3.5-4B \\
        --renderer-name qwen3 \\
        --log-path ~/logs/dpo_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent.parent.parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

import tinker_dpo  # noqa: E402
from weight.train.formatter import WeightDPODataBuilder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training (weight-format)")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--renderer-name", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--load-checkpoint-path", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    dataset_builder = WeightDPODataBuilder(
        train_path=args.train_path,
        test_path=args.test_path,
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=args.model_name,
            renderer_name=args.renderer_name,
            max_length=args.max_length,
            batch_size=args.batch_size,
        ),
    )

    config = tinker_dpo.Config(
        log_path=args.log_path,
        model_name=args.model_name,
        dataset_builder=dataset_builder,
        renderer_name=args.renderer_name,
        learning_rate=args.learning_rate,
        dpo_beta=args.dpo_beta,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        load_checkpoint_path=args.load_checkpoint_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        max_steps=args.max_steps,
    )

    tinker_dpo.main(config)


if __name__ == "__main__":
    main()
