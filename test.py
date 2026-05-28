#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET_LAYERS = (1, 3, 5, 7)


def import_runtime_dependencies():
    global torch, cudnn, dist, DataLoader, tqdm
    global StreamingDatasetAdapter, collate_bypass_tuples, model_forward_unscale
    global LayeredDepth, init_model, get_config_from_path, setup_distributed, to_cuda

    import torch
    import torch.backends.cudnn as cudnn
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from dataset import StreamingDatasetAdapter, collate_bypass_tuples, model_forward_unscale
    from dataset.layereddepth import LayeredDepth
    from model import init_model
    from util.config import get_config_from_path
    from util.dist import setup_distributed
    from util.train import to_cuda


def parse_args():
    parser = argparse.ArgumentParser(description="Run SeeGroup on the LayeredDepth test split.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "val.py")
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "predictions" / "layereddepth_test")
    parser.add_argument("--format", choices=("npy", "png"), default="npy")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument(
        "--mask-invalid-for-save",
        action="store_true",
        help="Zero out pixels rejected by the predicted validity mask in saved predictions.",
    )
    mask_group.add_argument(
        "--no-mask",
        dest="mask_invalid_for_save",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(mask_invalid_for_save=False)
    parser.add_argument("--png-scale", type=float, default=1000.0)
    parser.add_argument("--png-compression", type=int, default=3)
    return parser.parse_args()


def repo_path(path):
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def load_config(args):
    config = get_config_from_path(args.config.expanduser().as_posix())
    if args.overrides is not None:
        with open(args.overrides.expanduser(), "r") as f:
            config.merge_from_dict(json.load(f))

    checkpoint_path = args.checkpoint_path or config.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("test.py requires --checkpoint-path or checkpoint_path in the config.")

    checkpoint_path = repo_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    config["checkpoint_path"] = checkpoint_path.as_posix()
    config["resumed_from"] = checkpoint_path.as_posix()
    config["cli"] = "test"
    return config


def sample_index(sample, fallback):
    index = sample.get("index", fallback)
    if isinstance(index, (list, tuple)):
        index = index[0] if index else fallback
    if hasattr(index, "numel") and hasattr(index, "flatten"):
        index = index.flatten()[0].item() if index.numel() > 0 else fallback

    text = str(index)
    if text.isdigit():
        return str(int(text))
    return text.replace(os.sep, "_").replace(" ", "_")


def depth_to_uint16(depth, scale):
    import numpy as np

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.clip(depth * float(scale), 0, np.iinfo(np.uint16).max)
    return depth.astype(np.uint16)


def save_depth(path, depth, output_format, png_scale, png_compression):
    path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    depth = depth.astype(np.float32, copy=False)
    if output_format == "npy":
        np.save(path.with_suffix(".npy"), depth)
        return
    import cv2

    cv2.imwrite(
        path.with_suffix(".png").as_posix(),
        depth_to_uint16(depth, png_scale),
        [cv2.IMWRITE_PNG_COMPRESSION, png_compression],
    )


def save_prediction(output_dir, index, depth, args):
    png_compression = max(0, min(9, int(args.png_compression)))
    for layer_idx, layer_label in enumerate(TARGET_LAYERS):
        save_depth(output_dir / f"{index}_{layer_label}", depth[layer_idx], args.format, args.png_scale, png_compression)


def run_predictions(model, dataloader, output_dir, args, rank):
    model.eval()
    iterator = enumerate(dataloader)
    progress = None
    if rank == 0:
        progress = tqdm(desc="Predicting LayeredDepth test", unit="sample")

    try:
        with torch.no_grad():
            for batch_idx, sample in iterator:
                index = sample_index(sample, batch_idx)

                sample = to_cuda(sample)
                result = model_forward_unscale(model, {"image": sample["image"]})
                raw_depth = result["depth"][0]
                valid_mask = result.get("valid_mask", None)
                valid_mask = valid_mask[0] > 0 if valid_mask is not None else None

                depth_for_sort = raw_depth
                if valid_mask is not None and args.mask_invalid_for_save:
                    depth_for_sort = depth_for_sort.masked_fill(~valid_mask, 100000.0)
                depth, _ = depth_for_sort.sort(dim=0)
                if valid_mask is not None and args.mask_invalid_for_save:
                    depth = depth.masked_fill(depth >= 100000.0, 0.0)

                save_prediction(
                    output_dir=output_dir,
                    index=index,
                    depth=depth.detach().cpu().numpy(),
                    args=args,
                )
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()


def main():
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import_runtime_dependencies()
    config = load_config(args)
    rank, world_size = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    config["local_rank"] = local_rank
    config["rank"] = rank
    config["world_size"] = world_size

    cudnn.enabled = True
    cudnn.benchmark = True

    model = init_model(config)
    dataset = LayeredDepth(
        mode="test",
        hf_dataset="princeton-vl/LayeredDepth",
        hf_split="test",
        streaming=True,
        cache_dir=args.cache_dir.expanduser().as_posix() if args.cache_dir else None,
    )
    dataloader_dataset = StreamingDatasetAdapter(dataset)

    dataloader = DataLoader(
        dataloader_dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=collate_bypass_tuples,
    )

    run_predictions(model, dataloader, output_dir, args, rank)

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        print(f"Saved predictions to {output_dir}")


if __name__ == "__main__":
    main()
