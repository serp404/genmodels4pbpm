import numpy as np
import typing as tp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
import pm4py
import math

from logmentations.datasets.base import EmbeddedEvent


def prediction_collate_fn(
    prefix_sample: tp.List[tp.Tuple[tp.List[EmbeddedEvent], EmbeddedEvent]]
) -> tp.Tuple[FloatTensor, LongTensor, FloatTensor]:

    activity_batch = torch.tensor(
        [item[1].act for item in prefix_sample],
        dtype=torch.int64
    )

    timestamps_batch = torch.tensor(
        [item[1].time for item in prefix_sample],
        dtype=torch.float32
    )

    prefix_batch = nn.utils.rnn.pad_sequence(
        [torch.tensor([e.emb for e in item[0]], dtype=torch.float32)
            for item in prefix_sample],
        batch_first=True,
        padding_value=0.
    )

    return prefix_batch, activity_batch, timestamps_batch


def generation_collate_fn(
    traces: tp.List[tp.List[EmbeddedEvent]]
) -> tp.Tuple[LongTensor, FloatTensor, FloatTensor]:

    activities = nn.utils.rnn.pad_sequence(
        [torch.tensor([e.act for e in item], dtype=torch.int64)
            for item in traces],
        batch_first=True,
        padding_value=0
    )

    timestamps = nn.utils.rnn.pad_sequence(
        [torch.tensor([e.time for e in item], dtype=torch.float32)
            for item in traces],
        batch_first=True,
        padding_value=-1.
    )

    embeddings = nn.utils.rnn.pad_sequence(
        [torch.tensor([e.emb for e in item], dtype=torch.float32)
            for item in traces],
        batch_first=True,
        padding_value=0.
    )

    return activities, timestamps, embeddings


def time_aware_data_split(
    log: pm4py.objects.log.obj.EventLog, sizes: tp.Tuple[float, float, float]
) -> tp.Tuple[
    pm4py.objects.log.obj.EventLog,
    pm4py.objects.log.obj.EventLog,
    pm4py.objects.log.obj.EventLog
]:

    assert len(sizes) == 3, "Sizes must be of size 3"
    assert all([s >= 0 for s in sizes]), "Part size must be greater than 0"
    assert sum(sizes) == 1., "Part sizes must sum to 1"

    starts = [trace[0]["time:timestamp"].timestamp() for trace in log]
    train_bound = np.quantile(starts, sizes[0])
    val_bound = np.quantile(starts, sizes[0] + sizes[1])

    train = pm4py.filter_log(
        lambda t: t[0]["time:timestamp"].timestamp() <= train_bound, log
    )

    val = pm4py.filter_log(
        lambda t:
            train_bound < t[0]["time:timestamp"].timestamp() <= val_bound,
        log
    )

    test = pm4py.filter_log(
        lambda t: val_bound < t[0]["time:timestamp"].timestamp(), log
    )

    return train, val, test


def reparametrize(mu: FloatTensor, logvar: FloatTensor) -> FloatTensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

@torch.no_grad()
def get_grad_norm(model, norm_type=2):
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
        ),
        norm_type,
    )
    return total_norm.item()

def compute_mask(input_ix: LongTensor, eos_ix: int) -> LongTensor:
    return F.pad(torch.cumsum(input_ix == eos_ix, dim=-1)[..., :-1] < 1, pad=(1, 0, 0, 0), value=True)

def ohe2gumble(x: LongTensor, smooth_prob=0.9, tau=0.001) -> FloatTensor:
    x_smoothed = torch.where(x == 1, smooth_prob, (1. - smooth_prob) / (x.shape[-1] - 1))
    return nn.functional.gumbel_softmax(x_smoothed, tau=tau)

def kld_weight_annealing(n: int, max_n: int, max_value: float = 1.) -> float:
    return (math.exp(n / max_n) - 1) / (math.exp(1) - 1) * max_value

def uniform_kl(class_probs: np.array) -> float:
    '''
        Implements KL(uniform || class_probs) computation
    '''

    assert np.allclose(np.sum(class_probs), 1.), "The input probabilities do not sum to 1"
    p = 1. / len(class_probs)
    return np.sum(p * np.log(p / class_probs))
