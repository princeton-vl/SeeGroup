#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from model import init_model
from loss import init_criterion
from dataset import (
    evaluate_layereddepth,
    get_dataset_name,
    init_dataloader,
)
from engine import init_optimizer_scheduler

from util.config import get_config_from_path
from util.train import to_cuda
from util.log import init_wandb, broadcast_wandb_dir, setup_logger, wandb_log_scalars
from util.dist import setup_distributed


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description='Run SeeGroup training.')
    parser.add_argument('--config', type=Path, default=ROOT / 'config' / 'train.py')
    parser.add_argument('--overrides', type=Path, default=None)
    parser.add_argument('--wandb-mode', default='offline')
    return parser.parse_args()


def load_config(args):
    config = get_config_from_path(args.config.expanduser().as_posix())
    if args.overrides is not None:
        with open(args.overrides.expanduser(), 'r') as f:
            config.merge_from_dict(json.load(f))
    return config


def set_seed(seed, deterministic=True, rank=0):
    seed = int(seed) + int(rank)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)

def main():
    args = parse_args()
    os.environ.setdefault('WANDB_MODE', args.wandb_mode)

    config = load_config(args)
    rank, world_size = setup_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    config['local_rank'] = local_rank
    config['rank'] = rank
    config['world_size'] = world_size

    deterministic = config.get('deterministic', False)
    if 'seed' in config and config['seed'] is not None:
        set_seed(config['seed'], deterministic=deterministic, rank=rank)

    cudnn.enabled = True
    cudnn.benchmark = not deterministic

    if rank == 0:
        init_wandb(config)
    broadcast_wandb_dir(config)
    logger = setup_logger(config['log_dir'], rank)

    train(config, logger)

def train(config, logger):
    model = init_model(config)
    train_loader = init_dataloader(config, 'train')
    val_loaders = init_dataloader(config, 'val')
    steps_per_epoch = resolve_steps_per_epoch(config, train_loader)
    total_steps = config['epochs'] * steps_per_epoch

    criterion = init_criterion(config, total_steps=total_steps)
    optimizer, scheduler = init_optimizer_scheduler(config, model, total_steps=total_steps)

    step = 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    logger.info(f'Training for {config["epochs"]} epochs, {steps_per_epoch} steps per epoch, {config["bs"] * world_size} samples per step')
    
    for epoch in range(config['epochs']):
        set_dataloader_epoch(train_loader, epoch + 1)

        if epoch == 0 and config['validate_before_training']:
            validate(logger, model, val_loaders, step, config)

        model.train()
        torch.cuda.reset_peak_memory_stats()
        for batch_idx, inputs in enumerate(train_loader):
            if batch_idx >= steps_per_epoch:
                break

            optimizer.zero_grad()
            inputs = to_cuda(inputs)

            outputs = model(inputs)

            loss = criterion(inputs, outputs)
            loss['total'].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

            optimizer.step()
            scheduler.step()
            criterion.step()

            if config.get('rank', config['local_rank']) == 0 and step % config['save_loss_interval'] == 0:
                logger.info(f'loss {loss}')
                wandb_log_scalars(loss, step, 'train')
                wandb_log_scalars({'lr': scheduler.get_last_lr()[0]}, step, 'train')

                grad_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item()
                wandb_log_scalars({'grad_norm': grad_norm}, step, 'train')

            if 'validation_interval_step' in config and step % config['validation_interval_step'] == 0:
                validate(logger, model, val_loaders, step, config)

            step += 1

        if epoch % config['validation_interval'] == 0:
            validate(logger, model, val_loaders, step, config)

        if config.get('rank', config['local_rank']) == 0 and epoch % config['save_checkpoint_interval'] == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }
            save_path = os.path.join(config['log_dir'], 'checkpoints', f'{epoch}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(checkpoint, save_path)
        if dist.is_initialized():
            dist.barrier()


def set_dataloader_epoch(dataloader, epoch):
    if hasattr(dataloader.dataset, 'set_epoch'):
        dataloader.dataset.set_epoch(epoch)


def resolve_steps_per_epoch(config, train_loader):
    if config.get('steps_per_epoch') is not None:
        return int(config['steps_per_epoch'])

    num_examples = getattr(train_loader.dataset, 'num_examples', None)
    if num_examples is not None:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        return math.ceil(int(num_examples) / (world_size * int(config['bs'])))

    raise ValueError(
        'Streaming training requires config["steps_per_epoch"] or a dataset '
        'num_examples value.'
    )

@torch.no_grad()
def validate(logger, model, val_loaders, step, config):
    model.eval()

    logger.info('Start validating')
    for val_loader in val_loaders:
        dataset_name = get_dataset_name(val_loader.dataset)
        if dataset_name == 'LayeredDepth':
            metrics = evaluate_layereddepth(model, val_loader)
        else:
            raise ValueError(f'Unsupported validation dataset: {dataset_name}')

        metrics = {f'{dataset_name}_{key}': value for key, value in metrics.items()}
        if config.get('rank', config['local_rank']) == 0:
            wandb_log_scalars(metrics, step, 'val')
            logger.info(f'Validation metrics: {metrics}')

if __name__ == '__main__':
    main()
