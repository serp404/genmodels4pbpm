import torch
import torch.nn as nn
import typing as tp
import numpy as np

from logmentations.training.configs import BaseConfig
from logmentations.utils.utils import get_grad_norm
from logmentations.losses.loss import trace_recontruction_loss, latent_kld_loss

def train_predictive_epoch(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    config: BaseConfig
) -> tp.Tuple[float, float, float, float]:

    DEVICE = config.device
    optimizer = config.optimizer

    criterion_ce = nn.CrossEntropyLoss(ignore_index=0)
    criterion_mae = nn.L1Loss()

    act_weight, time_weight = config.act_weight, config.time_weight
    losses, losses_ce, losses_mae = [], [], []
    grad_norms = []

    model.train()
    for prefix, acts, times in data:
        prefix, acts, times = prefix.to(DEVICE), acts.to(DEVICE), times.to(DEVICE)
        logits, pred_times = model.forward(prefix)
        loss_act = criterion_ce(logits, acts)

        if model.predict_time:
            loss_time = criterion_mae(pred_times, times)
        else:
            loss_time = torch.tensor(0.)

        loss = act_weight * loss_act + time_weight * loss_time

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_value)
        optimizer.step()

        losses.append(loss.item())
        losses_ce.append(loss_act.item())
        losses_mae.append(loss_time.item())
        grad_norms.append(get_grad_norm(model))

    return np.mean(losses), np.mean(losses_ce), np.mean(losses_mae), np.mean(grad_norms)


def train_generative_epoch(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    config: BaseConfig
) -> tp.Tuple[float, float, float, float, float]:

    DEVICE = config.device
    optimizer = config.optimizer
    loss_weights = config.loss_weights
    act_weight, time_weight = config.act_weight, config.time_weight
    kld_weight = config.kld_weight

    losses, act_losses, time_losses, kld_losses = 0., 0., 0., 0.
    grad_norms = 0.
    n_samples = 0

    model.train()
    for acts, times, embs in data:
        logit_acts, logit_times, mu, logvar = model.forward(embs.to(DEVICE))

        act_loss, time_loss = trace_recontruction_loss(
            acts[:, 1:].to(DEVICE), logit_acts,
            times[:, 1:].to(DEVICE), logit_times.squeeze(dim=2),
            class_weights=loss_weights,
            ignore_index=0
        )

        kld_loss = latent_kld_loss(mu, logvar)
        loss = act_weight * act_loss + time_weight * time_loss + kld_weight * kld_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_value)
        optimizer.step()

        losses += loss.item()
        act_losses += act_loss.item()
        time_losses += time_loss.item()
        kld_losses += kld_loss.item()
        grad_norms += get_grad_norm(model)
        n_samples += embs.shape[0]

    return losses / n_samples, act_losses / n_samples, time_losses / n_samples, \
        kld_losses / n_samples, grad_norms / n_samples
