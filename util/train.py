def to_cuda(batch):
    if isinstance(batch, dict):
        return {k: to_cuda(v) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [to_cuda(x) for x in batch]
    elif isinstance(batch, tuple):
        return tuple(to_cuda(x) for x in batch)
    elif hasattr(batch, "cuda"):
        return batch.cuda(non_blocking=True)
    return batch
