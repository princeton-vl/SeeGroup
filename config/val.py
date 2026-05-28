from mmengine.config import read_base
with read_base():
    from .dataset.layereddepth import layereddepth_dataset
    from .model.seegroup import seegroup_model

project_name = 'seegroup'
base_log_dir = 'runs'
experiment_name = 'eval'

cli = 'val'

num_workers=2

checkpoint_path = 'checkpoints/seegroup.pth'
load_mappings=[]
model=seegroup_model

datasets={
    'val': [layereddepth_dataset],
}
