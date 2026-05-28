import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.dataloader import default_collate

from util.train import to_cuda

from dataset.layereddepth_syn import LayeredDepthSyn
from dataset.layereddepth import LayeredDepth, layereddepth_tuple_correct, get_layer_name, get_is_fake

from mmengine.registry import DATASETS

__all__ = ['LayeredDepthSyn', 'LayeredDepth', 'get_dataset_name']

mp.set_sharing_strategy('file_system')


class StreamingDatasetAdapter(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_name = dataset.__class__.__name__
        self.num_examples = getattr(dataset, 'num_examples', None)

    def __iter__(self):
        worker = get_worker_info()
        samples = self.dataset.iter_samples()
        if worker is None:
            yield from samples
            return

        for index, sample in enumerate(samples):
            if index % worker.num_workers == worker.id:
                yield sample

    def set_epoch(self, epoch):
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)


def is_streaming_dataset(dataset):
    return bool(getattr(dataset, 'streaming', False))


def has_iter_samples(dataset):
    return callable(getattr(dataset, 'iter_samples', None))


def wrap_streaming_dataset(dataset):
    if is_streaming_dataset(dataset) or has_iter_samples(dataset):
        return StreamingDatasetAdapter(dataset)
    return dataset


def get_dataset_name(dataset):
    return getattr(dataset, 'dataset_name', dataset.__class__.__name__)


def model_forward_unscale(model, sample):
    model_for_forward = model.module if hasattr(model, 'module') else model
    return model_for_forward.forward_unscale(sample)

def collate_bypass_tuples(batch):
    bypass_keys = ['pairs', 'trips', 'quads']
    keys = set().union(*(sample.keys() for sample in batch))
    out = {}

    for key in keys:
        values = [sample[key] for sample in batch]
        if key in bypass_keys:
            out[key] = values
        else:
            try:
                out[key] = default_collate(values)
            except:
                out[key] = values
    return out


def init_dataloader(config, split):
    datasets = [DATASETS.build(config_dataset) for config_dataset in config['datasets'][split]]
    num_workers = config.get('num_workers', 10)

    if split == 'train':
        if len(datasets) != 1:
            raise ValueError('Training expects a single dataset.')
        dataset = datasets[0]
        dataloader = DataLoader(
            wrap_streaming_dataset(dataset),
            batch_size=config['bs'],
            pin_memory=True,
            num_workers=num_workers,
        )
        return dataloader

    else:
        dataloaders = []
        
        for dataset in datasets:
            dataloader = DataLoader(
                wrap_streaming_dataset(dataset),
                batch_size=1,
                pin_memory=True,
                num_workers=num_workers,
                collate_fn=collate_bypass_tuples,
            )
            dataloaders.append(dataloader)
        
        return dataloaders


class MetricTracker:
    def __init__(self):
        self.metrics = {}
        self.reset()
    
    def reset(self):
        self.metrics = {}
    
    def update(self, metrics):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = {
                    'sum': 0,
                    'count': 0
                }
            self.metrics[key]['sum'] += value
            self.metrics[key]['count'] += 1

    def get_sum(self):
        return {key: self.metrics[key]['sum'] for key in self.metrics}

    def get_count(self):
        return {key: self.metrics[key]['count'] for key in self.metrics}
    
    def get_average(self):
        return {key: self.metrics[key]['sum'] / self.metrics[key]['count'] for key in self.metrics}
    
def aggregate_values_distributed(values):
    # Make sure all ranks reduce the same set of keys in the same order
    world_size = dist.get_world_size()

    local_keys = list(values.keys())
    gathered_keys = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_keys, local_keys)

    # Union of all keys across ranks, sorted for deterministic order
    all_keys = sorted({k for keys in gathered_keys for k in keys})

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vec = torch.tensor([float(values.get(k, 0.0)) for k in all_keys], device=device)

    # Single collective over the aligned vector
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)

    return {k: v.item() for k, v in zip(all_keys, vec)}

def aggregate_metrics_distributed(metric_tracker):
    sum_dict = metric_tracker.get_sum()
    count_dict = metric_tracker.get_count()
    
    sum_dict = aggregate_values_distributed(sum_dict)
    count_dict = aggregate_values_distributed(count_dict)
    
    return {key: sum_dict[key] / count_dict[key] for key in sum_dict}

def evaluate_layereddepth(
    model,
    val_loader,
    max_samples=None,
):
    data_iter = enumerate(val_loader)

    metric_tracker = MetricTracker()

    for i, sample in data_iter:
        if max_samples is not None and i >= max_samples:
            break

        sample = to_cuda(sample)
        img = sample['image']

        result = model_forward_unscale(model, {'image': img})
        raw_depth = result['depth'][0]

        mask = None
        mask_torch = None
        if 'valid_mask' in result:
            mask_torch = result['valid_mask'][0] > 0.01
        elif 'weight' in result:
            mask_torch = result['weight'][0] > 0.01
        if mask_torch is not None:
            mask = mask_torch.detach().cpu().numpy()

        raw_np = raw_depth.detach().cpu().numpy()

        for tuple_name in ['pairs', 'trips', 'quads']:
            if tuple_name not in sample or len(sample[tuple_name]) == 0:
                continue

            tuples = sample[tuple_name][0]
            if len(tuples) == 0:
                continue

            for single_tuple in tuples:
                correctness = int(layereddepth_tuple_correct(single_tuple, raw_np, mask))
                layer = get_layer_name(single_tuple)
                is_fake = get_is_fake(single_tuple)

                metric_tracker.update({f'{tuple_name}/acc': correctness})
                metric_tracker.update({f'{tuple_name}/{layer}': correctness})

                if is_fake:
                    metric_tracker.update({f'{tuple_name}/fake': correctness})
                    metric_tracker.update({f'{tuple_name}/{layer}_fake': correctness})
                else:
                    metric_tracker.update({f'{tuple_name}/real': correctness})
                    metric_tracker.update({f'{tuple_name}/{layer}_real': correctness})

    if dist.is_initialized():
        metrics = aggregate_metrics_distributed(metric_tracker)
    else:
        metrics = metric_tracker.get_average()

    if not metrics:
        raise ValueError(
            'LayeredDepth evaluation produced no tuple metrics. The dataset split '
            'likely has no pairs/trips/quads annotations; use the Hugging Face '
            'validation split for benchmark evaluation.'
        )

    return metrics
