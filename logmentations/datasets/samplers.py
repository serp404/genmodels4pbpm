import torch
import random


class LengthAwareSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(
        self, data_len: int, batch_size: int, group_size: int
    ) -> None:

        self.data_len = data_len
        self.batch_size = batch_size
        self.group_size = group_size
        self.id2group = torch.zeros(data_len)
        self.n_groups = data_len // group_size

        # assume that input dataset is already sorted by prefix length
        for k in range(self.n_groups):
            lower_bound = k * self.group_size
            upper_bound = (k + 1) * self.group_size
            self.id2group[lower_bound:upper_bound] = k

    def __iter__(self):
        num_yielded = 0
        indices = torch.arange(self.data_len)
        while num_yielded < self.data_len:
            current_group = random.randint(0, self.n_groups - 1)
            current_group_size = torch.sum(self.id2group == current_group)
            random_ids = torch.randperm(current_group_size)[:self.batch_size]
            batch_ids = indices[self.id2group == current_group][random_ids]
            num_yielded += len(batch_ids)
            yield batch_ids.tolist()

    def __len__(self) -> int:
        return self.data_len // self.batch_size
