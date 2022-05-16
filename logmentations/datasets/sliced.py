import torch
import typing as tp
import tqdm

from logmentations.datasets.base import LogsDataset, EmbeddedEvent
from logmentations.datasets.base import SIMPLE_TRACE, SIMPLE_EVENT


class SlicedLogsDataset(torch.utils.data.Dataset):
    def __init__(self, logs: LogsDataset) -> None:
        super().__init__()
        self.sliced_data: tp.List[
            tp.Tuple[tp.List[EmbeddedEvent], EmbeddedEvent]
        ] = []

        for trace in tqdm.tqdm(logs, "Slicing logs into model pairs"):
            for end_pos in range(1, len(trace) - 1):
                self.sliced_data.append(
                    (trace[:end_pos], trace[end_pos])
                )

        # sorting data by length for efficient sampling
        self.sliced_data.sort(key=lambda x: len(x[0]))

    def __len__(self) -> int:
        return len(self.sliced_data)

    def __getitem__(
        self, idx: int
    ) -> tp.Tuple[tp.List[EmbeddedEvent], EmbeddedEvent]:
        return self.sliced_data[idx]

    def extend_prefixes(
        self, extra_data: tp.List[tp.Tuple[SIMPLE_TRACE, SIMPLE_EVENT]]
    ) -> None:
        pass
