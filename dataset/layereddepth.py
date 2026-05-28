import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from dataset.hf_utils import (
    get_row_value,
    image_to_float_rgb,
    load_hf_dataset,
    split_hf_stream_by_rank,
)
from dataset.transform import NormalizeImage, PrepareForNet

from mmengine.registry import DATASETS

@DATASETS.register_module()
class LayeredDepth(Dataset):
    def __init__(
        self,
        mode='test',
        size=(1024, 1024),
        hf_dataset='princeton-vl/LayeredDepth',
        hf_split=None,
        streaming=True,
        cache_dir=None,
        tuple_layer='layer_all',
        num_examples=None,
    ):
        self.mode = mode
        self.size = size
        self.hf_dataset = hf_dataset
        self.hf_split = hf_split or ('validation' if mode == 'val' else 'test')
        self.streaming = streaming
        self.cache_dir = cache_dir
        self.tuple_layer = tuple_layer
        self.num_examples = num_examples
        self.transform = {
            'val': Compose([
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]),
            'test': Compose([
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]),
        }

        self.dataset = self._load_hf_dataset()

    def _load_hf_dataset(self):
        return load_hf_dataset(
            self.hf_dataset,
            self.hf_split,
            streaming=self.streaming,
            cache_dir=self.cache_dir,
        )

    def set_epoch(self, epoch):
        pass

    def _normalize_tuple_point(self, point, h, w):
        x, y, layer = [int(v) for v in point]
        if layer not in (1, 3, 5, 7):
            return None
        layer = (layer - 1) // 2

        if x < 0 or x >= w or y < 0 or y >= h or layer < 0 or layer > 3:
            return None
        return [x, y, layer]

    def _load_hf_tuples(self, row, h, w):
        tuple_data = get_row_value(row, ['tuples.json', 'tuples'])
        if self.tuple_layer not in tuple_data:
            raise KeyError(f'Hugging Face tuple data has no "{self.tuple_layer}" section.')

        tuple_group = tuple_data[self.tuple_layer]
        parsed = {'pairs': [], 'trips': [], 'quads': []}

        for tuple_type in ['pairs', 'trips', 'quads']:
            for single_tuple in tuple_group.get(tuple_type, []):
                points = []
                for point_key in ['p1', 'p2', 'p3', 'p4']:
                    if point_key not in single_tuple:
                        continue
                    point = self._normalize_tuple_point(
                        single_tuple[point_key],
                        h,
                        w,
                    )
                    if point is None:
                        points = []
                        break
                    points.append(point)

                if not points:
                    continue

                is_valid = bool(single_tuple.get('is_real', single_tuple.get('is_valid', True)))
                points.append([is_valid, is_valid, is_valid])
                parsed[tuple_type].append(tuple(points))

        return parsed

    def _normalize_hf_index(self, index):
        if index is None:
            return index
        text = str(index)
        if text.isdigit():
            return str(int(text))
        return text

    def build_hf_sample(self, row, item=None):
        image = image_to_float_rgb(get_row_value(row, ['image.png', 'image', 'rgb']))
        index = self._normalize_hf_index(row.get('__key__', item))

        sample = {'image': image, 'index': index}
        tuples = {}
        if self.mode != 'test':
            tuples = self._load_hf_tuples(row, image.shape[0], image.shape[1])
        sample = self.transform[self.mode](sample)
        sample.update(tuples)
        return sample
        
    def __getitem__(self, item):
        raise TypeError('LayeredDepth is iterable and does not support indexing.')

    def iter_samples(self):
        for item, row in enumerate(split_hf_stream_by_rank(self.dataset)):
            yield self.build_hf_sample(row, item)

    def __len__(self):
        raise TypeError('LayeredDepth does not expose a per-rank length.')

def get_unique_depths(depths):
    new_depths = []
    for d in depths:
        if len(new_depths) == 0:
            new_depths.append(d)
        else:
            if abs(d - new_depths[-1]) > 0.02:
                new_depths.append(d)
    return new_depths

def get_depth(pd, x, y, l, mask=None):
    pred_depths = pd[:, y, x]
    
    if mask is not None:
        pred_mask = mask[:, y, x]
        pred_depths = pred_depths[pred_mask > 0.01]
    # print(pred_mask)
    # print(pred_depths)

    pred_depths = [d for d in pred_depths if d > 0.02]
    pred_depths = sorted(list(set(pred_depths)))
    # print('before unique', pred_depths)
    pred_depths = get_unique_depths(pred_depths)
    # print('after unique', pred_depths)
    if len(pred_depths) <= l:
        return None
    return pred_depths[l]

def get_layer_name(single_tuple):
    current_layer = None
    *points, is_valid = single_tuple
    for point in points:
        x, y, l = point
        if current_layer is None:
            current_layer = l
        elif current_layer != l:
            return 'mixed'
    assert current_layer is not None
    return str(int(current_layer))

def get_is_fake(single_tuple):
    *points, is_valid = single_tuple
    return not is_valid[0]

def layereddepth_tuple_correct(single_tuple, pd, mask=None, verbose=False):
    *points, is_valid = single_tuple
    is_valid = is_valid[0]

    last_depth = None
    for point in points:
        x, y, l = point
        d = get_depth(pd, x, y, l, mask)
        if is_valid:
            if d is None: 
                return False
            elif last_depth is not None and last_depth >= d:
                return False
            last_depth = d
        else:
            if d is not None:
                return False
    return True
