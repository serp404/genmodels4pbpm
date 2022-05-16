import torch
import torch.nn as nn
import typing as tp
import numpy as np
import tqdm
from sklearn.metrics import f1_score, accuracy_score

from logmentations.training.configs import BaseConfig
from logmentations.utils.utils import compute_mask
from logmentations.metrics.metrics import cer_score

@torch.no_grad()
def eval_predictive_model(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    config: BaseConfig
) -> tp.Tuple[float, float, float, float, float, float]:

    DEVICE = config.device

    criterion_ce = nn.CrossEntropyLoss(ignore_index=0)
    criterion_mae = nn.L1Loss()
    act_weight, time_weight = config.act_weight, config.time_weight

    model.eval()
    losses, losses_ce, losses_mae = [], [], []
    total_predicts, total_truth = [], []

    for prefix, acts, times in data:
        prefix, acts, times = prefix.to(DEVICE), acts.to(DEVICE), times.to(DEVICE)
        logits, pred_times = model.forward(prefix)

        loss_act = criterion_ce(logits, acts)

        if model.predict_time:
            loss_time = criterion_mae(pred_times, times)
        else:
            loss_time = torch.tensor(0.)

        loss = act_weight * loss_act + time_weight * loss_time
        
        losses.append(loss.item())
        losses_ce.append(loss_act.item())
        losses_mae.append(loss_time.item())

        total_predicts.extend(torch.argmax(logits, dim=1).cpu().tolist())
        total_truth.extend(acts.cpu().tolist())
        
    accuracy = accuracy_score(total_truth, total_predicts)
    f1_macro = f1_score(total_truth, total_predicts, average='macro')

    return np.mean(losses), np.mean(losses_ce), np.mean(losses_mae), \
        accuracy, f1_macro


@torch.no_grad()
def eval_generative_model(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    config: BaseConfig
) -> tp.Tuple[float, float, float, float]:

    DEVICE = config.device

    model.eval()
    min_accuracy, mae, cer, length_mae = 0., 0., 0., 0.
    n_samples = 0

    for acts, times, embs in data:
        src_acts, src_times, src_embs = acts[:, 1:].to(DEVICE), times[:, 1:].to(DEVICE), embs.to(DEVICE)
        starts = src_embs[:, :1]

        latents, _, _, _ = model.encode(src_embs)
        acts_predicts, time_predicts = model.autoregressive_decode(latents, starts)

        common_length = min(src_acts.shape[1], acts_predicts.shape[1])
        src_mask = compute_mask(src_acts, eos_ix=2)
        pred_mask = compute_mask(acts_predicts, eos_ix=2)

        src_lens = src_mask.sum(dim=1)
        pred_lens = pred_mask.sum(dim=1)

        length_mae += torch.abs(src_lens - pred_lens).sum().item()
        n_samples += len(acts)

        cer_scores = torch.tensor([
            cer_score(tgt[:tgt_len], pred[:pred_len])
            for tgt, pred, tgt_len, pred_len in zip(src_acts, acts_predicts, src_lens, pred_lens)
        ])

        src_acts = src_acts[:, :common_length]
        src_times = src_times[:, :common_length]

        pred_acts = acts_predicts[:, :common_length]
        pred_times = time_predicts[:, :common_length]

        accuracy_scores = torch.sum((src_acts == pred_acts) * src_mask[:, :common_length], dim=1) / src_mask[:, :common_length].sum(dim=1)
        mae_scores = torch.sum(torch.abs((src_times - pred_times) * src_mask[:, :common_length]), dim=1) / src_mask[:, :common_length].sum(dim=1)

        min_accuracy += accuracy_scores.sum().item()
        mae += mae_scores.sum().item()
        cer += cer_scores.sum().item()

    return min_accuracy / n_samples, cer / n_samples, mae / n_samples, length_mae / n_samples


def eval_prediction_test_metrics(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    config: BaseConfig,
    time2days: tp.Callable[[float], float],
    n_runs: int = 20,
) -> tp.Tuple[float, float, float, float, float]:

    test_losses = []
    test_ces = []
    test_maes = []
    test_accuracies = []
    test_f1_scores = []

    for _ in tqdm.tqdm(list(range(n_runs))):
        test_loss, test_ce, test_mae, test_accuracy, test_f1_macro = eval_predictive_model(
            model, data, config
        )

        test_losses.append(test_loss)
        test_ces.append(test_ce)
        test_maes.append(test_mae)
        test_accuracies.append(test_accuracy)
        test_f1_scores.append(test_f1_macro)

    return np.mean(test_losses), np.mean(test_ces), time2days(np.mean(test_maes)), \
        np.mean(test_accuracies), np.mean(test_f1_scores)
