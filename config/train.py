from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from engine import scheduler_exp
from loss import Loss, IntensityLoss, LayeredGradMatchingScaleLoss

from mmengine.config import read_base
with read_base():
    from .dataset.layereddepth_syn import layereddepth_syn_train
    from .dataset.layereddepth import layereddepth_dataset
    from .model.seegroup import seegroup_model

project_name = 'seegroup'
base_log_dir = 'runs'
experiment_name = 'train'

cli = 'train'

bs=1
epochs=200
num_workers=10
steps_per_epoch=None

validation_interval=1
save_checkpoint_interval=3
save_loss_interval=20 # 100 steps

max_grad_norm=0.5

validate_before_training = True

pretrained_from='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth'
load_mappings=[
    ('multi_depth_module.depth_head', 'depth_head'),
]
freeze_modules=[]

model=seegroup_model

datasets={
    'train': [layereddepth_syn_train],
    'val': [layereddepth_dataset],
}

criterion=dict(
    type=Loss,
    losses=[
        dict(
            loss_config=dict(
                type=IntensityLoss,
                alignment='normalization',
                align_all_layers=True,
                gamma=0.1,
            ),
            loss_input=dict(
                ds=('OUTPUT', 'depth'),
                bs=('OUTPUT', 'b1'),
                target=('INPUT', 'depth'),
                target_mask=('INPUT', 'valid_mask'),
            ),
            loss_weight=1.0
        ),
        dict(
            loss_config=dict(
                type=LayeredGradMatchingScaleLoss,
                scale_level=4,
                alignment='normalization',
                align_all_layers=True,
                sort_pred=True,
                sort_targ=True,
                layer_weights=[1.2, 1.0, 1.0, 1.0],
            ),
            loss_input=dict(
                pred=('OUTPUT', 'depth'),
                targ=('INPUT', 'depth'),
                pred_mask=('OUTPUT', 'valid_mask'),
                targ_mask=('INPUT', 'valid_mask'),
            ),
            loss_weight=1.0
        )
    ]
)

optimizer=dict(
    type=AdamW,
    params=[
        {'params': 'pretrained', 'lr_scale': 0.1},
    ],
    lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

scheduler=dict(
    type=LambdaLR,
    lr_lambda=scheduler_exp,
)
