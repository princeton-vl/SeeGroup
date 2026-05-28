import torch

from engine.scheduler_function import scheduler_exp

from mmengine.registry import OPTIMIZERS, PARAM_SCHEDULERS

__all__ = ['scheduler_exp']

def get_model_target_parameters(model, target_name):
    return [param for name, param in model.named_parameters() if target_name in name]

def get_model_rest_parameters(model, target_names):
    return [param for name, param in model.named_parameters() if not any(target_name in name for target_name in target_names)]

def init_optimizer_scheduler(config, model, **kwargs):
    if 'params' in config['optimizer']:
        target_names = []
        for group in config['optimizer']['params']:
            target_names.append(group['params'])
            group['params'] = get_model_target_parameters(model, group['params'])
            group['lr'] = config['optimizer']['lr'] * group['lr_scale']
        config['optimizer']['params'].append({'params': get_model_rest_parameters(model, target_names), 'lr': config['optimizer']['lr']})
    else:
        config['optimizer']['params'] = [{'params': model.parameters(), 'lr': config['optimizer']['lr']}]

    optimizer = OPTIMIZERS.build(config['optimizer'])
    config['scheduler']['optimizer'] = optimizer
    config['scheduler']['lr_lambda'] = config['scheduler']['lr_lambda'](**kwargs)
    scheduler = PARAM_SCHEDULERS.build(config['scheduler'])

    if 'resume_from' in config:
        checkpoint = torch.load(config['resume_from'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    return optimizer, scheduler
