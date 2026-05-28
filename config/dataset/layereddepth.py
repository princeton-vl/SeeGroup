from dataset import LayeredDepth

hf_cache_dir = None

layereddepth_dataset=dict(
    type=LayeredDepth,
    mode='val',
    hf_dataset='princeton-vl/LayeredDepth',
    hf_split='validation',
    streaming=True,
    cache_dir=hf_cache_dir,
    tuple_layer='layer_all',
    num_examples=300,
)
