#!/usr/bin/env python3
"""Download Tinker LoRA checkpoints and merge them into one LoRA adapter."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunInfo:
    slug: str
    config_path: Path
    checkpoint_path: Path
    base_model: str
    sampler_path: str
    adapter_dir: Path


def load_local_env() -> None:
    for env_path in (
        Path.cwd() / ".env",
        Path.cwd() / "scripts" / "weight" / ".env",
    ):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip("'").strip('"')


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_runs(model_weights_dir: Path, work_dir: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for ckpt_path in sorted(model_weights_dir.glob("*/checkpoints.jsonl")):
        run_dir = ckpt_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config beside {ckpt_path}")

        config = _read_json(config_path)
        checkpoints = _read_jsonl(ckpt_path)
        if not checkpoints:
            raise ValueError(f"No checkpoints in {ckpt_path}")

        final_rows = [row for row in checkpoints if row.get("name") == "final"]
        checkpoint = final_rows[-1] if final_rows else checkpoints[-1]
        sampler_path = checkpoint.get("sampler_path")
        if not isinstance(sampler_path, str) or not sampler_path.startswith("tinker://"):
            raise ValueError(f"Bad sampler_path in {ckpt_path}: {sampler_path!r}")

        base_model = config.get("model_name")
        if not isinstance(base_model, str) or not base_model:
            raise ValueError(f"Missing model_name in {config_path}")

        runs.append(
            RunInfo(
                slug=run_dir.name,
                config_path=config_path,
                checkpoint_path=ckpt_path,
                base_model=base_model,
                sampler_path=sampler_path,
                adapter_dir=work_dir / "adapters" / run_dir.name,
            )
        )
    if not runs:
        raise FileNotFoundError(f"No checkpoints.jsonl found under {model_weights_dir}")
    return runs


def validate_base_model(runs: list[RunInfo], requested_base_model: str | None) -> str:
    base_models = sorted({run.base_model for run in runs})
    if requested_base_model:
        return requested_base_model
    if len(base_models) != 1:
        raise ValueError(
            "Runs use different base models; pass --base-model explicitly: "
            + ", ".join(base_models)
        )
    return base_models[0]


def write_manifest(runs: list[RunInfo], base_model: str, work_dir: Path) -> Path:
    manifest = {
        "base_model": base_model,
        "runs": [
            {
                "slug": run.slug,
                "sampler_path": run.sampler_path,
                "adapter_dir": str(run.adapter_dir),
                "config_path": str(run.config_path),
                "checkpoint_path": str(run.checkpoint_path),
            }
            for run in runs
        ],
    }
    path = work_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def download_adapters(runs: list[RunInfo], force: bool) -> None:
    try:
        from tinker_cookbook import weights
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "Could not import tinker_cookbook.weights. Install with:\n"
            "  uv pip install --python SkyRL/.venv/bin/python tinker tinker-cookbook\n"
            "and export TINKER_API_KEY before running --download."
        ) from exc

    for run in runs:
        if force and run.adapter_dir.exists():
            shutil.rmtree(run.adapter_dir)
        if (run.adapter_dir / "adapter_config.json").exists():
            print(f"skip existing adapter: {run.slug} -> {run.adapter_dir}")
            continue
        run.adapter_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"download {run.slug}: {run.sampler_path}")
        weights.download(tinker_path=run.sampler_path, output_dir=str(run.adapter_dir))


def _load_adapter_state(adapter_dir: Path) -> dict[str, Any]:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))
    if bin_path.exists():
        import torch

        return torch.load(str(bin_path), map_location="cpu")
    raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin in {adapter_dir}")


def _adapter_scale(config: dict[str, Any]) -> float:
    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}
    if rank_pattern or alpha_pattern:
        raise ValueError("rank_pattern/alpha_pattern adapters need per-module handling")

    rank = int(config["r"])
    alpha = float(config.get("lora_alpha", rank))
    if config.get("use_rslora"):
        return alpha / math.sqrt(rank)
    return alpha / rank


def merge_lora_adapters(
    runs: list[RunInfo],
    base_model: str,
    output_dir: Path,
    weights: list[float] | None,
    normalize: bool,
) -> None:
    import torch
    from safetensors.torch import save_file

    if weights is None:
        weights = [1.0 / len(runs)] * len(runs)
    if normalize:
        total = sum(weights)
        if total == 0:
            raise ValueError("Cannot normalize weights that sum to zero")
        weights = [weight / total for weight in weights]

    configs = [_read_json(run.adapter_dir / "adapter_config.json") for run in runs]
    states = [_load_adapter_state(run.adapter_dir) for run in runs]
    scales = [_adapter_scale(config) for config in configs]

    first_keys = set(states[0])
    for run, state in zip(runs[1:], states[1:], strict=True):
        if set(state) != first_keys:
            missing = sorted(first_keys - set(state))[:5]
            extra = sorted(set(state) - first_keys)[:5]
            raise ValueError(f"Adapter key mismatch for {run.slug}; missing={missing}, extra={extra}")

    output_state: dict[str, Any] = {}
    a_keys = sorted(key for key in first_keys if "lora_A" in key)
    handled: set[str] = set()
    for a_key in a_keys:
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in first_keys:
            raise ValueError(f"Missing B matrix for {a_key}")

        a_parts = []
        b_parts = []
        for state, weight, scale in zip(states, weights, scales, strict=True):
            a = state[a_key].detach().cpu()
            b = state[b_key].detach().cpu()
            a_parts.append(a)
            b_parts.append(b * float(weight) * float(scale))

        output_state[a_key] = torch.cat(a_parts, dim=0).contiguous()
        output_state[b_key] = torch.cat(b_parts, dim=1).contiguous()
        handled.add(a_key)
        handled.add(b_key)

    for key in sorted(first_keys - handled):
        output_state[key] = states[0][key].detach().cpu()

    out_config = dict(configs[0])
    total_rank = sum(int(config["r"]) for config in configs)
    out_config["r"] = total_rank
    out_config["lora_alpha"] = total_rank
    out_config["use_rslora"] = False
    out_config["base_model_name_or_path"] = base_model
    out_config["inference_mode"] = True

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adapter_config.json").write_text(
        json.dumps(out_config, indent=2) + "\n", encoding="utf-8"
    )
    save_file(output_state, str(output_dir / "adapter_model.safetensors"))
    (output_dir / "README.md").write_text(
        "# Linear merged LoRA adapter\n\n"
        f"Base model: `{base_model}`\n\n"
        "This adapter is an exact linear merge of LoRA deltas. The rank is the "
        "sum of input ranks, so it can be served as one PEFT/SGLang/vLLM LoRA.\n",
        encoding="utf-8",
    )


def parse_weights(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(part) for part in value.split(",") if part.strip()]


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-weights-dir", default="model-weights")
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Scratch directory for downloaded adapters and the merged output.",
    )
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--weights", default=None, help="Comma-separated linear weights")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--lora-output", default=None)
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    runs = discover_runs(Path(args.model_weights_dir), work_dir)
    base_model = validate_base_model(runs, args.base_model)
    weights = parse_weights(args.weights)
    normalize = not args.no_normalize

    manifest_path = write_manifest(runs, base_model, work_dir)
    print(f"manifest: {manifest_path}")
    print(f"base model: {base_model}")
    for run in runs:
        print(f"{run.slug}: {run.sampler_path} -> {run.adapter_dir}")

    if args.download:
        download_adapters(runs, force=args.force)

    output_dir = Path(args.lora_output or (work_dir / "merged_lora_linear")).resolve()
    merge_lora_adapters(runs, base_model, output_dir, weights, normalize)
    print(f"merged LoRA adapter: {output_dir}")


if __name__ == "__main__":
    main()
