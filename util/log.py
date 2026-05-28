import os
import wandb
import logging

import torch.distributed as dist
from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log

def setup_logger(log_dir, rank, save_to_file=True, color=True, abbrev_name=None):
    logger = logging.getLogger(f'train_logger_{rank}')
    logger.handlers.clear()  # Clear any existing handlers
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logging.getLogger("wandb").setLevel(logging.ERROR)
    
    plain_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    if rank == 0:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        if color:
            if abbrev_name is None:
                abbrev_name = ""
            color_formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=logger.name,
                abbrev_name=abbrev_name
            )
            stream_handler.setFormatter(color_formatter)
        else:
            stream_handler.setFormatter(plain_formatter)
        logger.addHandler(stream_handler)
    
    if save_to_file:
        file_handler = logging.FileHandler(os.path.join(log_dir, f'rank{rank}.log'))
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)
    
    return logger

def make_config_serializable(config_obj):
    if isinstance(config_obj, dict):
        return {k: make_config_serializable(v) for k, v in config_obj.items()}
    elif isinstance(config_obj, list):
        return [make_config_serializable(item) for item in config_obj]
    elif isinstance(config_obj, (int, float, str, bool)) or config_obj is None:
        return config_obj
    else:
        return str(config_obj)

def init_wandb(config):
    safe_config = make_config_serializable(config._cfg_dict)
    wandb.init(
        project=config['project_name'],
        dir=config['base_log_dir'],
        name=config['experiment_name'],
        config=safe_config,
    )
    config['log_dir'] = wandb.run.dir
    print('wandb_dir', config['log_dir'])
    config.dump(os.path.join(config['log_dir'], 'config.py'))

def broadcast_wandb_dir(config):
    if not dist.is_available() or not dist.is_initialized():
        return
    wandb_dir_list = [config.get('log_dir', None)]
    dist.broadcast_object_list(wandb_dir_list, src=0)
    config['log_dir'] = wandb_dir_list[0]

def wandb_log_scalars(scalars: dict, step: int, split='train'):
    for name, value in scalars.items():
        print(f'{split}/{name}: {value}')
        wandb.log({f'{split}/{name}': value}, step=step)
