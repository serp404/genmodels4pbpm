import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
import typing as tp

from logmentations.utils import reparametrize


class LogVAE(nn.Module):
    def __init__(
        self, n_features: int, latent_dim: int, num_classes: int, emb_dim: int = 64,
        hid_dim: int = 128, num_layers: int = 1, bidirectional: bool = False
    ) -> None:

        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.init_dim = num_layers * 2 if bidirectional else num_layers

        self.init_layers = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=emb_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(normalized_shape=emb_dim)
        )

        self.encoder_rnn = nn.LSTM(
            input_size=emb_dim,
            num_layers=num_layers,
            hidden_size=hid_dim,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.mu_layer = nn.Sequential(
            nn.Linear(in_features=hid_dim * self.init_dim, out_features=latent_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(normalized_shape=latent_dim)
        )

        self.logvar_layer = nn.Sequential(
            nn.Linear(in_features=hid_dim * self.init_dim, out_features=latent_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(normalized_shape=latent_dim)
        )

        self.decoder_rnn = nn.LSTM(
            input_size=emb_dim,
            num_layers=num_layers,
            hidden_size=hid_dim,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.latent_hs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=latent_dim, out_features=hid_dim),
                nn.LeakyReLU(),
                nn.LayerNorm(normalized_shape=hid_dim)
            ) for _ in range(self.init_dim)
        ])

        self.latent_cs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=latent_dim, out_features=hid_dim),
                nn.LeakyReLU(),
                nn.LayerNorm(normalized_shape=hid_dim)
            ) for _ in range(self.init_dim)
        ])

        self.act_layers = nn.Linear(hid_dim, num_classes)
        self.time_layers = nn.Linear(hid_dim, 1)

    def encode(
        self, x: FloatTensor
    ) -> tp.Tuple[FloatTensor, FloatTensor, FloatTensor]:
        self.encoder_rnn.flatten_parameters()

        model_device = next(self.parameters()).device
        batch_size = x.shape[0]
        init_dim = self.num_layers

        x = self.init_layers(x)

        if self.bidirectional:
            init_dim *= 2

        h0 = torch.ones(
            (init_dim, batch_size, self.hid_dim),
            device=model_device
        )

        c0 = torch.zeros(
            (init_dim, batch_size, self.hid_dim),
            device=model_device
        )

        _, (hn, _) = self.encoder_rnn(x, (h0, c0))
        hidden = hn.permute(1, 2, 0).flatten(start_dim=1)

        mu, logvar = self.mu_layer(hidden), self.logvar_layer(hidden)
        latent = reparametrize(mu, logvar)
        return latent, mu, logvar, x

    def _decode(
        self, x: FloatTensor, latent: FloatTensor
    ) -> tp.Tuple[FloatTensor, FloatTensor]:
        self.decoder_rnn.flatten_parameters()

        seq_len = x.shape[1]
        h0, c0 = self._init_hiddens(latent)
 
        hiddens_list = []
        for prefix_len in range(1, seq_len):
            _, (hn, _) = self.decoder_rnn(
                x[:, :prefix_len].contiguous(), (h0, c0)
            )
            hiddens_list.append(hn[-1])
        outs = torch.stack(hiddens_list, dim=1)

        acts = self.act_layers(outs)
        ts = self.time_layers(outs)
        return acts, ts

    def forward(
        self, x: FloatTensor
    ) -> tp.Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor]:

        latent, mu, logvar, x = self.encode(x)
        activities, times = self._decode(x, latent)
        return activities, times, mu, logvar

    def _init_hiddens(self, latent: FloatTensor) -> tp.Tuple[FloatTensor, FloatTensor]:
        h = torch.stack(
            [mapping(latent) for mapping in self.latent_hs],
            dim=0
        ).contiguous()

        c = torch.stack(
            [mapping(latent) for mapping in self.latent_cs],
            dim=0
        ).contiguous()

        return h, c

    @torch.no_grad()
    def autoregressive_decode(
        self, latent: FloatTensor, init_x: FloatTensor, max_length: int = 150
    ) -> tp.Tuple[FloatTensor, FloatTensor]:
        self.decoder_rnn.flatten_parameters()

        x = self.init_layers(init_x)
        h, c = self._init_hiddens(latent)

        pred_acts = []
        pred_times = []

        for _ in range(max_length):
            with torch.no_grad():
                _, (hn, _) = self.decoder_rnn(x, (h, c))
                hidden = hn[-1]

                next_a = torch.argmax(self.act_layers(hidden), dim=1)
                next_t = self.time_layers(hidden)

            pred_acts.append(next_a)
            pred_times.append(next_t.squeeze(dim=1))

            next_emb = torch.cat((F.one_hot(next_a, num_classes=self.num_classes), next_t), dim=1)
            x = torch.cat((x, torch.unsqueeze(self.init_layers(next_emb), dim=1)), dim=1)

        pred_acts_total = torch.stack(pred_acts, dim=1)
        pred_times_total = torch.stack(pred_times, dim=1)
        return pred_acts_total, pred_times_total
