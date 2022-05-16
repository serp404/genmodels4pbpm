from __future__ import annotations
import numpy as np
import typing as tp
import random
import tqdm
from sklearn.neighbors import KernelDensity

from logmentations.datasets.base import LogsDataset


class StatisticsAugmentation:
    def __init__(self, act2id: tp.Dict[str, int]) -> None:
        self.act2id = act2id
        self.transition_distributions: tp.Dict[int, tp.Dict[str, tp.Any]] = {}

    def _make_kde(self, data: tp.List[float]) -> KernelDensity:
        npdata = np.expand_dims(np.array(data, dtype=float), axis=1)
        return KernelDensity().fit(npdata)

    def _sample_activity(self, from_act_id: int):
        random_idx = random.randint(
            0, len(self.transition_distributions[from_act_id]["acts"]) - 1
        )
        return self.transition_distributions[from_act_id]["acts"][random_idx]

    def _sample_time(self, from_act_id: int, to_act_id: int) -> float:
        time_kdes = self.transition_distributions[from_act_id]["times"]
        return abs(time_kdes[to_act_id].sample(1)[0][0])

    def _increase_distribution(
        self, from_act_id: int, to_act_id: int, time_diff: float
    ) -> None:

        if from_act_id not in self.transition_distributions:
            self.transition_distributions[from_act_id] = {
                "acts": [],  # list of possible next act_ids
                "times": {}  # dict of possible ts diffs for every next act_id
            }

        self.transition_distributions[from_act_id]["acts"].append(to_act_id)
        time_kdes = self.transition_distributions[from_act_id]["times"]

        if from_act_id == self.act2id["<BOS>"] or \
                to_act_id == self.act2id["<EOS>"]:

            if to_act_id not in time_kdes:
                time_kdes[to_act_id] = [0.]
        else:

            if to_act_id not in time_kdes:
                time_kdes[to_act_id] = []
            time_kdes[to_act_id].append(time_diff)

    def fit(self, log: LogsDataset) -> StatisticsAugmentation:
        if len(self.transition_distributions) > 0:
            self.transition_distributions = {}

        for trace in tqdm.tqdm(log, desc="Processing transitions"):
            from_act_id = self.act2id["<BOS>"]
            from_ts = trace[0]["time:timestamp"].timestamp()

            for event in trace:
                to_act_id = self.act2id[event["concept:name"]]
                to_ts = event["time:timestamp"].timestamp()
                self._increase_distribution(
                    from_act_id, to_act_id, to_ts - from_ts
                )
                from_act_id, from_ts = to_act_id, to_ts

            self._increase_distribution(from_act_id, self.act2id["<EOS>"], 0.)

        for from_act_id, stats in tqdm.tqdm(
                self.transition_distributions.items(), desc="Making KDEs"
        ):
            for to_act_id, times in stats["times"].items():
                stats["times"][to_act_id] = self._make_kde(times)

        return self

    def sample(self):
        from_act_id = self._sample_activity(self.act2id["<BOS>"])
        from_ts = 0.

        trace = [(from_act_id, from_ts)]
        while from_act_id != 2:
            to_act_id = self._sample_activity(from_act_id)
            to_ts = self._sample_time(from_act_id, to_act_id)

            trace.append((to_act_id, to_ts))
            from_act_id = to_act_id

        return trace[:-1]
