import torch
import torch.nn as nn
import typing as tp


# RNN model based on vanilla LSTM architecture
class LstmModel(nn.Module):
    def __init__(
        self, vocab_size: int, n_features: int,
        emb_size: int = 128, hid_size: int = 64, num_layers: int = 1,
        bidirectional: bool = False, predict_time: bool = False
    ) -> None:

        super().__init__()
        self.vocab_size = vocab_size
        self.n_features = n_features
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.predict_time = predict_time

        self.init_layers = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=emb_size),
            nn.ReLU(),
        )

        self.rnn_layers = nn.LSTM(
            input_size=emb_size,
            num_layers=num_layers,
            hidden_size=hid_size,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.logit_layers = nn.Sequential(
            nn.Linear(in_features=hid_size, out_features=hid_size),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(in_features=hid_size, out_features=hid_size),
            nn.Dropout(p=0.4),
            nn.ReLU()
        )

        self.activity_layer = nn.Linear(
            in_features=hid_size, out_features=vocab_size
        )

        if predict_time:
            self.time_layer = nn.Linear(
                in_features=hid_size, out_features=1
            )

    def forward(
        self, prefix_batch: torch.FloatTensor
    ) -> tp.Tuple[torch.FloatTensor, tp.Optional[torch.FloatTensor]]:

        model_device = next(self.parameters()).device
        batch_size = prefix_batch.shape[0]

        input_batch = self.init_layers(prefix_batch)
        init_dim = self.num_layers
        if self.bidirectional:
            init_dim *= 2

        h0 = torch.randn(
            (init_dim, batch_size, self.hid_size),
            device=model_device
        )

        c0 = torch.ones(
            (init_dim, batch_size, self.hid_size),
            device=model_device
        )

        _, (hn, _) = self.rnn_layers(input_batch, (h0, c0))
        logits = self.logit_layers(hn[init_dim - 1])

        activity_output = self.activity_layer(logits)
        time_output = None

        if self.predict_time:
            time_output = self.time_layer(logits)

        return activity_output, time_output
