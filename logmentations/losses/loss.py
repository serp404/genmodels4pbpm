import torch
import torch.nn.functional as F
from torch import Tensor
import typing as tp

def trace_recontruction_loss(
    real_acts: Tensor, gen_logit_acts: Tensor,
    real_times: Tensor, gen_times: Tensor,
    ignore_index: tp.Optional[int] = None,
    class_weights: tp.Optional[Tensor] = None
) -> tp.Tuple[Tensor, Tensor]:

    mask = 1.
    n_samples = real_acts.size
    if ignore_index is not None:
        mask = (real_acts != ignore_index).float()
        n_samples = mask.sum()

    act_loss = F.cross_entropy(
        gen_logit_acts.transpose(1, 2), real_acts,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction='sum'
    )

    time_loss = F.l1_loss(gen_times * mask, real_times * mask, reduction='sum')
    return act_loss / n_samples, time_loss / n_samples

def latent_kld_loss(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
