import torch
import torch.distributed as dist
from mmengine.registry import MODELS

from model.seegroup import SeeGroup, SeeGroupDepthHead, SeeGroupMultiDepthHead

__all__ = [
    'SeeGroup',
    'SeeGroupDepthHead',
    'SeeGroupMultiDepthHead',
]

def summarize_keys(keys):
    return keys

def normalize_release_type_names(config):
    if isinstance(config, list):
        return [normalize_release_type_names(item) for item in config]
    if isinstance(config, tuple):
        return tuple(normalize_release_type_names(item) for item in config)
    if not isinstance(config, dict):
        return config

    normalized = {}
    for key, value in config.items():
        if key == 'type' and isinstance(value, str) and value.startswith('model.'):
            normalized[key] = value.split('.')[-1]
        else:
            normalized[key] = normalize_release_type_names(value)
    return normalized

def load_mappings(model_dict, load_mappings):
    new_model_dict = model_dict.copy()
    for key, value in model_dict.items():
        for dst_key, src_key in load_mappings:
            if src_key in key:
                new_key = key.replace(src_key, dst_key)
                new_model_dict[new_key] = value
    return new_model_dict

def init_model(config):
    model = MODELS.build(normalize_release_type_names(config['model']))

    if 'pretrained_from' in config and config['pretrained_from'] is not None:
        model_dict = torch.load(config['pretrained_from'], map_location='cpu')
        if 'model' in model_dict:
            model_dict = model_dict['model']
            new_model_dict = dict()
            for key, value in model_dict.items():
                key_components = key.split('.')
                if key_components[0] == 'module':
                    key_components.pop(0)
                new_key = '.'.join(key_components)
                new_model_dict[new_key] = value
            model_dict = new_model_dict
        if 'load_mappings' in config:
            model_dict = load_mappings(model_dict, config['load_mappings'])
        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

        if config['local_rank'] == 0:
            if len(missing_keys) > 0:
                print(f'Missing keys: {summarize_keys(missing_keys)}')
            if len(unexpected_keys) > 0:
                print(f'Unexpected keys: {summarize_keys(unexpected_keys)}')
    elif 'resumed_from' in config:
        checkpoint = torch.load(config['resumed_from'], map_location='cpu')
        model_dict = checkpoint['model']
        new_model_dict = dict()
        for key, value in model_dict.items():
            key_components = key.split('.')
            if key_components[0] == 'module':
                key_components.pop(0)
            new_key = '.'.join(key_components)
            new_model_dict[new_key] = value
        model_dict = new_model_dict
        if 'load_mappings' in config:
            model_dict = load_mappings(model_dict, config['load_mappings'])

        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
        if len(missing_keys) > 0:
            print(f'Missing keys: {summarize_keys(missing_keys)}')
        if len(unexpected_keys) > 0:
            print(f'Unexpected keys: {summarize_keys(unexpected_keys)}')

    if config['cli'] in ['train', 'val', 'eval', 'test']:
        model.cuda(config['local_rank'])
        if dist.is_available() and dist.is_initialized():
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[config['local_rank']],
                broadcast_buffers=False,
                output_device=config['local_rank'],
                find_unused_parameters=True,
            )
    else:
        model.cuda()

    if 'freeze_modules' in config and len(config['freeze_modules']) > 0:
        for name, param in model.named_parameters():
            if any(module in name for module in config['freeze_modules']):
                param.requires_grad = False

    return model
