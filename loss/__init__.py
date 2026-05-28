from torch import nn

from .gradient_matching import LayeredGradMatchingScaleLoss
from .intensity import IntensityLoss

from mmengine.registry import MODELS

__all__ = [
    'LayeredGradMatchingScaleLoss',
    'Loss',
    'IntensityLoss',
]


class Loss(nn.Module):
    def __init__(self, losses, total_steps=None):
        super().__init__()
        self.losses = {}
        self.current_step = 0
        self.total_steps = total_steps
        for loss_dict in losses:
            loss = MODELS.build(loss_dict['loss_config'])
            input_dict = loss_dict['loss_input']
            weight = loss_dict['loss_weight']
            self.losses[loss.__class__.__name__] = {
                'loss': loss,
                'input_dict': input_dict,
                'weight': weight,
            }

    def get_inputs(self, input_dict, inputs, outputs):
        loss_inputs = {}
        for key, value in input_dict.items():
            if value[0] == 'OUTPUT':
                loss_inputs[key] = outputs.get(value[1], None)
            elif value[0] == 'INPUT':
                loss_inputs[key] = inputs.get(value[1], None)
            else:
                raise ValueError(f'Invalid input type: {value[0]}')
        return loss_inputs

    def forward(self, inputs, outputs):
        loss_dict = {'total': 0}

        for loss_name, loss_info in self.losses.items():
            loss_inputs = self.get_inputs(loss_info['input_dict'], inputs, outputs)
            loss_value = loss_info['loss'](**loss_inputs)
            loss_dict[loss_name] = loss_value

            if isinstance(loss_info['weight'], list):
                weight = loss_info['weight'][0] + (
                    loss_info['weight'][1] - loss_info['weight'][0]
                ) * self.current_step / self.total_steps
            else:
                weight = loss_info['weight']
            loss_dict['total'] += loss_value * weight

        return loss_dict

    def step(self):
        self.current_step += 1


def init_criterion(config, total_steps=None):
    config['criterion']['total_steps'] = total_steps
    return MODELS.build(config['criterion'])
