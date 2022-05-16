from __future__ import annotations
import torch
import tqdm
import random
import typing as tp

from logmentations.datasets.base import LogsDataset
from logmentations.models.generation import LogVAE
from logmentations.utils import generation_collate_fn, compute_mask

class NeuralAugmentation:
    def __init__(
        self, model: LogVAE, act2id: tp.Dict[str, int],
        model_batch_size: int = 64, sample_size: int = 2500
    ) -> None:

        self.model = model
        self.device = next(model.parameters()).device
        self.batch_size = model_batch_size
        self.sample_size = sample_size
        self.act2id = act2id
        self.log_labels = None
        self.log_latents = None

    def fit(self, log: LogsDataset) -> NeuralAugmentation:
        self.log_labels, self.log_latents = [], []
        log_labels, traces_buffer = [], []

        subsample_ids = random.sample(list(range(len(log))), min(self.sample_size, len(log)))
        for i in tqdm.tqdm(subsample_ids, desc="Processing dataset"):
            activity_encoding = [0] * len(self.act2id)

            for event in log[i]:
                activity_encoding[event.act] += 1

            log_labels.append(activity_encoding)
            traces_buffer.append(log[i])

            if len(traces_buffer) == self.batch_size:
                _, _, embs = generation_collate_fn(traces_buffer)
                with torch.no_grad():
                    latents, _, _, _ = self.model.encode(embs.to(self.device))
                self.log_latents.append(latents.cpu())
                traces_buffer = []
        
        if len(traces_buffer) > 0:
            _, _, embs = generation_collate_fn(traces_buffer)
            with torch.no_grad():
                latents, _, _, _ = self.model.encode(embs.to(self.device))
            self.log_latents.append(latents.cpu())
            traces_buffer = []

        self.log_labels = torch.tensor(log_labels)
        self.log_latents = torch.cat(self.log_latents, dim=0)
        return self
    
    def sample(self, classes: tp.List[int], n_iters: int) -> tp.Any:
        assert self.log_labels is not None and self.log_latents is not None, "Augmentation model needs to be fitted."
        filter_t = (self.log_labels[:, classes].sum(dim=1) > 0)
        target_latents = self.log_latents[filter_t]

        coefs_ = torch.rand(n_iters)
        ids = torch.randint(high=len(target_latents), size=(n_iters, 2))

        dummy_starts = torch.zeros((self.batch_size, 1, len(self.act2id) + 1), device=self.device)
        dummy_starts[:, :, self.act2id["<BOS>"]] = 1.

        new_latents = []
        new_traces = []

        for i in tqdm.tqdm(range(n_iters), desc="Generating traces", total=n_iters):
            lat = coefs_[0] * target_latents[ids[i][0]] + (1. - coefs_[0]) * target_latents[ids[i][1]]
            new_latents.append(lat)

            if len(new_latents) == self.batch_size:
                batch = torch.stack(new_latents, dim=0).to(self.device)
                acts_predicts, time_predicts = self.model.autoregressive_decode(batch, dummy_starts)

                lengths = compute_mask(acts_predicts, eos_ix=self.act2id["<EOS>"]).sum(dim=1)
                for i in range(self.batch_size):
                    new_traces.append(list(zip(
                        acts_predicts[i, :lengths[i]-1].cpu().tolist(),
                        torch.relu(time_predicts[i, :lengths[i]-1]).cpu().tolist()
                    )))

                new_latents = []

        if len(new_latents) > 0:
            batch = torch.stack(new_latents, dim=0).to(self.device)
            acts_predicts, time_predicts = self.model.autoregressive_decode(batch, dummy_starts[:len(new_latents)])

            lengths = compute_mask(acts_predicts, eos_ix=self.act2id["<EOS>"]).sum(dim=1)
            for i in range(len(new_latents)):
                new_traces.append(list(zip(
                    acts_predicts[i, :lengths[i]-1].cpu().tolist(),
                    torch.relu(time_predicts[i, :lengths[i]-1]).cpu().tolist()
                )))

        return new_traces
