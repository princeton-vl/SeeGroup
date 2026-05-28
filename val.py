#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from dataset import (
    evaluate_layereddepth,
    get_dataset_name,
    init_dataloader,
)
from model import init_model
from util.config import get_config_from_path
from util.dist import setup_distributed


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SeeGroup validation.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "val.py")
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def repo_path(path: str | os.PathLike[str]) -> str:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved.as_posix()


def load_config(args: argparse.Namespace):
    config = get_config_from_path(args.config.expanduser().as_posix())
    if args.overrides is not None:
        with open(args.overrides.expanduser(), "r") as f:
            config.merge_from_dict(json.load(f))

    if args.checkpoint_path is not None:
        config["checkpoint_path"] = args.checkpoint_path.expanduser().as_posix()
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples

    return config


def main() -> None:
    args = parse_args()

    config = load_config(args)

    rank, _ = setup_distributed()
    cudnn.enabled = True
    cudnn.benchmark = True

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    config["local_rank"] = local_rank
    config["rank"] = rank

    evaluate(config)


def evaluate(config) -> None:
    checkpoint_path = config.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError('Validation requires config["checkpoint_path"] or --checkpoint-path.')

    checkpoint_path = repo_path(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    config["resumed_from"] = checkpoint_path
    print("Evaluating checkpoint", checkpoint_path)
    model = init_model(config)
    val_loaders = init_dataloader(config, "val")
    validate(model, val_loaders, config)


@torch.no_grad()
def validate(model, val_loaders, config) -> None:
    model.eval()

    for val_loader in val_loaders:
        dataset_name = get_dataset_name(val_loader.dataset)
        if dataset_name == "LayeredDepth":
            metrics = evaluate_layereddepth(
                model,
                val_loader,
                max_samples=config.get("max_samples"),
            )
        else:
            raise ValueError(f"Unsupported validation dataset: {dataset_name}")

        metrics_prefixed = {f"{dataset_name}_{key}": value for key, value in metrics.items()}
        if config.get("rank", config["local_rank"]) == 0:
            print(f"Validation metrics: {metrics_prefixed}")


if __name__ == "__main__":
    main()
