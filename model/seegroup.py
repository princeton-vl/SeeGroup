import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dinov2 import DINOv2
from model.util.blocks import _make_fusion_block, _make_r_scratch, _make_scratch
from model.util.transform import Resize

from mmengine.registry import MODELS


@MODELS.register_module()
class SeeGroupDepthHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=(256, 512, 1024, 1024),
    ):
        super().__init__()

        out_channels = list(out_channels)
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.r_projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        ])

        self.r_resize_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            nn.Identity(),
            nn.ConvTranspose2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        ])

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.r_scratch = _make_r_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1,
            head_features_1 // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Softplus(),
        )
        self.scratch.output_conv3 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Softplus(),
        )

    def _decode_depth_and_beta(self, out_features, patch_h, patch_w):
        out = []
        x_shapes = []
        for i, x in enumerate(out_features):
            x = x[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x_shapes.append(x.shape)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out,
            (int(patch_h * 14), int(patch_w * 14)),
            mode='bilinear',
            align_corners=True,
        )
        depth = self.scratch.output_conv2(out)
        beta = self.scratch.output_conv3(out)

        return depth, beta, (layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn), x_shapes

    def _update_features(self, out_features, refined_features, x_shapes, patch_h, patch_w):
        r_layer_1 = self.r_scratch.layer1_rn(refined_features[0])
        r_layer_2 = self.r_scratch.layer2_rn(refined_features[1])
        r_layer_3 = self.r_scratch.layer3_rn(refined_features[2])
        r_layer_4 = self.r_scratch.layer4_rn(refined_features[3])

        new_features = []
        for i, x in enumerate([r_layer_1, r_layer_2, r_layer_3, r_layer_4]):
            if i == 3:
                r_x = self.r_resize_layers[i](x, output_size=x_shapes[i])
            else:
                r_x = self.r_resize_layers[i](x)
            r_x = self.r_projects[i](r_x)
            r_x = r_x.reshape((r_x.shape[0], r_x.shape[1], patch_h * patch_w)).permute(0, 2, 1)

            new_feature = out_features[i][0] - r_x
            norm_old = torch.linalg.vector_norm(out_features[i][0], dim=1, keepdim=True)
            norm_new = torch.linalg.vector_norm(new_feature, dim=1, keepdim=True)
            new_feature = new_feature * (norm_old / (norm_new + 1e-6))
            new_feature = new_feature.clamp(min=0)
            new_features.append((new_feature, out_features[i][1]))

        return tuple(new_features)

    def forward(self, out_features, patch_h, patch_w):
        depth, beta, refined_features, x_shapes = self._decode_depth_and_beta(
            out_features,
            patch_h,
            patch_w,
        )
        new_features = self._update_features(
            out_features,
            refined_features,
            x_shapes,
            patch_h,
            patch_w,
        )
        return depth, beta, new_features


@MODELS.register_module()
class SeeGroupMultiDepthHead(nn.Module):
    def __init__(
        self,
        max_depth=20.0,
        num_heads=4,
        depth_head_module=None,
        min_beta=1.0,
        max_beta=10.0,
    ):
        super().__init__()

        if depth_head_module is None:
            raise ValueError('SeeGroupMultiDepthHead requires a depth_head_module.')

        self.max_depth = max_depth
        self.num_heads = num_heads
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.depth_head = MODELS.build(depth_head_module)

    def forward(self, features, patch_h, patch_w):
        depths = []
        betas = []

        for _ in range(self.num_heads):
            depth, beta, features = self.depth_head(features, patch_h, patch_w)
            depths.append(depth.squeeze(1) * self.max_depth)
            betas.append((beta.squeeze(1) + self.min_beta).clamp(min=self.min_beta, max=self.max_beta))

        depths = torch.stack(depths, dim=1)
        betas = torch.stack(betas, dim=1)

        return {
            'depth': depths,
            'b1': betas,
            'd2': None,
            'b2': None,
            'valid_mask': -torch.log(betas - self.min_beta),
        }


@MODELS.register_module()
class SeeGroup(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        multi_depth_module=None
    ):
        super(SeeGroup, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.multi_depth_module = MODELS.build(multi_depth_module)

    def forward(self, inputs):
        x = inputs['image']
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depths = self.multi_depth_module(features, patch_h, patch_w)

        return depths
    
    @torch.no_grad()
    def forward_unscale(self, inputs, input_size=518):
        raw_image = inputs['image']
        h, w = raw_image.shape[-2:]

        resizer = Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        )
        new_w, new_h = resizer.get_size(w, h)
        image = F.interpolate(raw_image, (new_h, new_w), mode="bilinear", align_corners=True)

        preds = self.forward({'image': image})
        rescaled_preds = {}

        for key, value in preds.items():
            new_value = []
            if value is None:
                continue
            for i in range(value.shape[1]):
                layered_value = value[:, i]
                layered_value = F.interpolate(layered_value[:, None], (h, w), mode="bilinear", align_corners=True)[:, 0]
                new_value.append(layered_value)
            new_value = torch.stack(new_value, dim=1)
            rescaled_preds[key] = new_value
        
        return rescaled_preds
