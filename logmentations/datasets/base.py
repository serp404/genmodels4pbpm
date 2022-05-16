import torch
import typing as tp
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import pm4py


SIMPLE_EVENT = tp.Tuple[int, float]
SIMPLE_TRACE = tp.List[SIMPLE_EVENT]
SIMPLE_LOG = tp.List[SIMPLE_TRACE]


# dimension num_classes
def process_activity(
    act: int, num_classes: int
) -> tp.List[int]:

    emb = [0] * num_classes
    emb[act] = 1
    return emb


def get_event_embedding(
    act: int, time: float, num_classes: int,
    time_normalizer: tp.Optional[tp.Callable[[float], float]] = None
) -> tp.List[float]:

    activity_embedding = process_activity(act, num_classes)
    time_embedding = [time if time_normalizer is None else time_normalizer(time)]
    return activity_embedding + time_embedding


class EmbeddedEvent:
    def __init__(
        self, act: int, time: float, emb: tp.List[float],
        init_time: tp.Optional[float] = None
    ) -> None:

        self.act = act
        self.time = time
        self.init_time = init_time
        self.emb = emb

    def __repr__(self) -> str:
        if self.init_time is None:
            return f"act_id: {self.act}, time: {self.time}"
        else:
            return f"act_id: {self.act}, time: {self.time}, init_time: {self.init_time}"


class LogsDataset(torch.utils.data.Dataset):
    def __init__(
        self, event_log: pm4py.objects.log.obj.EventLog,
        act2id: tp.Dict[str, int],
        time_applyer: tp.Callable[[float], float] = lambda x: x
    ) -> None:

        super().__init__()
        self.num_classes = len(act2id)
        self.act2id = act2id
        self.time_applyer = time_applyer

        self.embedded_log: tp.List[tp.List[EmbeddedEvent]] = []
        for trace in tqdm.tqdm(event_log, "Log dataset processing"):
            embedded_trace: tp.List[EmbeddedEvent] = []
            last_ts = trace[0]["time:timestamp"].timestamp()

            # BOS event
            embedded_trace.append(
                EmbeddedEvent(
                    act2id["<BOS>"],
                    time_applyer(0.),
                    get_event_embedding(
                        act2id["<BOS>"], 0.,
                        self.num_classes,
                        self.time_applyer
                    ), init_time=0.
                )
            )

            for event in trace:
                elapsed_time = event["time:timestamp"].timestamp() - last_ts
                embedded_trace.append(
                    EmbeddedEvent(
                        act2id[event["concept:name"]],
                        time_applyer(elapsed_time),
                        get_event_embedding(
                            act2id[event["concept:name"]],
                            elapsed_time,
                            self.num_classes, self.time_applyer
                        ),
                        init_time=elapsed_time
                    )
                )
                last_ts = event["time:timestamp"].timestamp()

            # EOS event
            embedded_trace.append(
                EmbeddedEvent(
                    act2id["<EOS>"],
                    time_applyer(0.),
                    get_event_embedding(
                        act2id["<EOS>"], 0.,
                        self.num_classes,
                        self.time_applyer
                    ),
                    init_time=0.
                )
            )
            self.embedded_log.append(embedded_trace)

    def __len__(self) -> int:
        return len(self.embedded_log)

    def __getitem__(self, idx: int) -> tp.List[EmbeddedEvent]:
        return self.embedded_log[idx]

    def extend_log(self, extra_data: SIMPLE_LOG) -> None:
        '''
            extra_data: pairs of act_id and normalized elapsed time
        '''

        for trace in tqdm.tqdm(extra_data, "Extending log"):
            embedded_trace: tp.List[EmbeddedEvent] = []

            # BOS event
            embedded_trace.append(
                EmbeddedEvent(
                    self.act2id["<BOS>"], 0.,
                    get_event_embedding(
                        self.act2id["<BOS>"], 0.,
                        self.num_classes
                    )
                )
            )

            for event in trace:
                embedded_trace.append(
                    EmbeddedEvent(
                        event[0], event[1],
                        get_event_embedding(
                            event[0], event[1],
                            self.num_classes
                        )
                    )
                )

            # EOS event
            embedded_trace.append(
                EmbeddedEvent(
                    self.act2id["<EOS>"], 0.,
                    get_event_embedding(
                        self.act2id["<EOS>"], 0.,
                        self.num_classes
                    )
                )
            )
            self.embedded_log.append(embedded_trace)

    def plot_activity_distribution(self) -> None:
        activities: tp.Dict[int, int] = {}
        for trace in self.embedded_log:
            for event in trace[1:-1]:
                activities[event.act] = activities.get(event.act, 0) + 1

        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=[p[0] for p in activities.items()],
            y=[p[1] for p in activities.items()]
        ).grid()
