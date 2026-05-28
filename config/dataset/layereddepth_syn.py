from dataset import LayeredDepthSyn

hf_cache_dir = None

layereddepth_syn_train=dict(
    type=LayeredDepthSyn,
    mode='train',
    hf_dataset='princeton-vl/LayeredDepth-Syn',
    hf_split='train',
    streaming=True,
    cache_dir=hf_cache_dir,
    shuffle_buffer_size=1024,
    num_examples=14800,
)
