from model import SeeGroup, SeeGroupDepthHead, SeeGroupMultiDepthHead


seegroup_model = dict(
    type=SeeGroup,
    encoder='vitl',
    multi_depth_module=dict(
        type=SeeGroupMultiDepthHead,
        max_depth=20,
        num_heads=4,
        min_beta=1.0,
        depth_head_module=dict(
            type=SeeGroupDepthHead,
            in_channels=1024,
            features=256,
            out_channels=[256, 512, 1024, 1024],
        ),
    ),
)
