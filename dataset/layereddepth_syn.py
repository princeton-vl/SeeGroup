import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.hf_utils import (
    depth_png_to_meters,
    get_row_value,
    image_to_float_rgb,
    load_hf_dataset,
    split_hf_stream_by_rank,
)
from dataset.transform import Crop, Flip, NormalizeImage, PrepareForNet, Resize

from mmengine.registry import DATASETS


def remove_none_keys(sample):
    return {k: v for k, v in sample.items() if v is not None}


@DATASETS.register_module()
class LayeredDepthSyn(Dataset):
    def __init__(
        self,
        mode='train',
        size=(518, 518),
        hf_dataset='princeton-vl/LayeredDepth-Syn',
        hf_split=None,
        streaming=True,
        cache_dir=None,
        shuffle_buffer_size=1024,
        seed=42,
        num_examples=None,
    ):
        self.mode = mode
        self.size = size
        self.hf_dataset = hf_dataset
        self.hf_split = hf_split or ('train' if mode == 'train' else 'validation')
        self.streaming = streaming
        self.cache_dir = cache_dir
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.epoch = 0
        self.num_examples = num_examples

        self.dataset = self._load_hf_dataset()

        net_w, net_h = size
        self.transform = {
            'train': Compose([
                Resize(
                    width=net_w,
                    height=net_h,
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                Crop(size[0]),
                Flip(p=0.5),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]),
            'val': Compose([
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]),
        }

    def _load_hf_dataset(self):
        return load_hf_dataset(
            self.hf_dataset,
            self.hf_split,
            streaming=self.streaming,
            cache_dir=self.cache_dir,
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def load_hf_data(self, row, key, data_type):
        if data_type == 'rgb':
            return image_to_float_rgb(get_row_value(row, ['image.png', 'image', 'rgb']))
        if data_type == 'depth':
            return depth_png_to_meters(get_row_value(row, [key, key.replace('_', '')]))
        raise ValueError(f'Unsupported data type: {data_type}')

    def postprocess_layered_depth(self, data_list):
        for current_layer in range(1, len(data_list)):
            for target_layer in range(current_layer):
                valid_current = data_list[current_layer] != 0
                valid_target = data_list[target_layer] != 0
                collapse_region = valid_current & (~valid_target)
                data_list[target_layer][collapse_region] = data_list[current_layer][collapse_region]
                data_list[current_layer][collapse_region] = 0

        return np.stack(data_list, axis=-1)

    def build_sample(self, image, depth_layered, index):
        depth_layered = self.postprocess_layered_depth(depth_layered)
        valid_mask = (depth_layered > 0).astype(np.float32)
        sample = {
            'index': index,
            'image': image,
            'depth': depth_layered,
            'valid_mask': valid_mask,
        }

        sample = remove_none_keys(sample)
        return self.transform[self.mode](sample)

    def build_hf_sample(self, row, item=None):
        image = self.load_hf_data(row, 'image.png', 'rgb')
        index = row.get('__key__', item)

        depth_layered = []
        for layer_id in (1, 3, 5, 7):
            depth_layered.append(self.load_hf_data(row, f'depth_{layer_id}.png', 'depth'))

        return self.build_sample(image, depth_layered, index)

    def __getitem__(self, item):
        raise TypeError('LayeredDepthSyn is iterable and does not support indexing.')

    def iter_samples(self):
        dataset = self.dataset
        if self.mode == 'train' and self.shuffle_buffer_size > 0:
            shuffle_kwargs = {'seed': self.seed + self.epoch}
            if self.streaming:
                shuffle_kwargs['buffer_size'] = self.shuffle_buffer_size
            dataset = dataset.shuffle(**shuffle_kwargs)
        for item, row in enumerate(split_hf_stream_by_rank(dataset)):
            yield self.build_hf_sample(row, item)

    def __len__(self):
        raise TypeError('LayeredDepthSyn does not expose a per-rank length.')
