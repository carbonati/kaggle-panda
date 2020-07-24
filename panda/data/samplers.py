from torch.utils.data import Sampler, DistributedSampler, Dataset
from operator import itemgetter
from utils.data_utils import stratify_batches


class BatchStratifiedSampler(Sampler):
    """Stratified batch sampler."""
    def __init__(self,
                 data_source,
                 indices,
                 labels,
                 batch_size,
                 drop_last=False,
                 random_state=6969):
        self.data_source = data_source
        self.indices = indices
        self.labels = labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_state = random_state

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        indices = stratify_batches(
            self.indices,
            self.labels,
            self.batch_size,
            drop_last=self.drop_last,
            random_state=self.random_state
        )
        return iter(indices)


class DatasetFromSampler(Dataset):
    """Dataset indices from a sampler."""
    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py"""
    def __init__(self, sampler, **kwargs):
        super(DistributedSamplerWrapper, self).__init__(DatasetFromSampler(sampler),
                                                        **kwargs)
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        return iter(itemgetter(*indexes_of_indexes)(self.dataset))

