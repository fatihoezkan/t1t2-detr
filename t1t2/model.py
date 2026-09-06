"""The network: a 64-point signal in, a fixed set of compartment candidates out.

DETR (Carion et al., 2020) with the image backbone replaced by an MLP over the signal.
Learned query vectors attend to the encoded signal through a transformer decoder, and each
query is read out into (T1, T2, weight) plus an existence logit. The model always emits
n_queries candidates; the Hungarian matcher in loss.py decides which count during training,
and a threshold on the existence score decides at evaluation time.
"""
from __future__ import annotations

import torch
from torch import nn


class SignalEncoder(nn.Module):
    """Four Linear/LayerNorm/ReLU blocks from the 64-point signal to one fs_dim vector.

    This is the counterpart of DETR's CNN backbone. A voxel signal is a short fixed-length
    vector, so one memory token is enough and no positional encoding is needed.
    """

    def __init__(self, input_dim: int, hidden_dim: int, fs_dim: int):
        super().__init__()
        # 64 -> hidden -> hidden -> hidden -> fs_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, fs_dim), nn.LayerNorm(fs_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MLPHead(nn.Module):
    """Three-layer MLP used for every prediction head."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # two hidden layers, linear output
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class T1T2DETR(nn.Module):
    """Signal in, set of compartments out.

    forward() returns (B, n_queries, 4) with [T1, T2, weight, existence_logit] per query.
    T1, T2 and weight are sigmoid outputs in the normalised [0, 1] space of the targets
    (see data.TargetNormalizer); existence is a raw logit for BCE-with-logits. With aux_loss
    on, forward() returns {"pred": final, "aux": [one prediction per decoder layer]}.
    """

    def __init__(self, cfg):
        super().__init__()
        # network sizes from the config
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim
        self.fs_dim = cfg.fs_dim
        self.n_queries = cfg.n_queries
        self.n_layers = cfg.n_dlayers
        self.n_heads = cfg.n_heads
        self.aux_loss = cfg.aux_loss

        # MLP backbone: signal -> one memory token
        self.encoder = SignalEncoder(self.input_dim, self.hidden_dim, self.fs_dim)

        # One learned vector per query. Nothing ties a query to a particular compartment.
        self.queries = nn.Embedding(self.n_queries, self.fs_dim)
        # transformer decoder: the queries attend to the memory token
        decoder_layer = nn.TransformerDecoderLayer(
            self.fs_dim, self.n_heads, dim_feedforward=self.hidden_dim, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.n_layers, norm=nn.LayerNorm(self.fs_dim)
        )

        # one regression head per output, applied to every query state
        self.t1_head = MLPHead(self.fs_dim, self.fs_dim, 1)
        self.t2_head = MLPHead(self.fs_dim, self.fs_dim, 1)
        self.w_head = MLPHead(self.fs_dim, self.fs_dim, 1)

        # Existence head. The two wirings differ in whether one query's decision can depend
        # on the other queries' states.
        self.exist_head_mode = cfg.exist_head
        if self.exist_head_mode not in {"joint", "shared"}:
            raise ValueError(
                f"model.exist_head must be joint|shared; got {self.exist_head_mode!r}"
            )

        if self.exist_head_mode == "joint":
            # Every query state is projected down to red_dim, the projections are
            # concatenated, and one head emits all n_queries logits at once, so it can
            # suppress a query that duplicates a neighbour.
            red_dim = self.fs_dim // 4
            self.qu_red = nn.Sequential(
                nn.Linear(self.fs_dim, red_dim), nn.LayerNorm(red_dim), nn.ReLU()
            )
            self.exist_head = MLPHead(red_dim * self.n_queries, self.fs_dim, self.n_queries)
        else:
            # One head with shared weights applied to each query state on its own, like the
            # T1/T2/weight heads. Nothing couples the queries.
            self.exist_head = MLPHead(self.fs_dim, self.fs_dim, 1)

    def forward(self, X):
        # encode the signal and tile the learned queries over the batch
        B = X.size(0)
        memory = self.encoder(X).unsqueeze(1)                     # (B, 1, fs_dim)
        hs = self.queries.weight.unsqueeze(0).expand(B, -1, -1)   # (B, n_queries, fs_dim)

        # Layers are stepped through by hand so that the intermediate states are reachable
        # for the auxiliary losses.
        aux = []
        for i in range(self.n_layers):
            hs = self.decoder.layers[i](tgt=hs, memory=memory)
            if self.aux_loss:
                aux.append(self._predict(hs))

        # final norm and read-out
        hs = self.decoder.norm(hs)
        out = self._predict(hs)
        if self.aux_loss:
            return {"pred": out, "aux": aux}
        return out

    def _predict(self, hs):
        """Run the heads over decoder states hs and return (B, n_queries, 4)."""
        # regression outputs in [0, 1], each (B, n_queries, 1)
        t1 = torch.sigmoid(self.t1_head(hs))
        t2 = torch.sigmoid(self.t2_head(hs))
        w = torch.sigmoid(self.w_head(hs))
        # Both wirings end at (B, n_queries, 1), so nothing downstream needs to know which
        # one was used.
        if self.exist_head_mode == "joint":
            hs_cat = self.qu_red(hs).reshape(hs.size(0), -1)   # (B, n_queries * red_dim)
            exist = self.exist_head(hs_cat).unsqueeze(2)       # (B, n_queries, 1)
        else:
            exist = self.exist_head(hs)                        # (B, n_queries, 1)
        return torch.cat([t1, t2, w, exist], dim=-1)  # (B, n_queries, 4)


def build_model(model_cfg) -> T1T2DETR:
    """Construct the model from a ModelConfig."""
    return T1T2DETR(model_cfg)
