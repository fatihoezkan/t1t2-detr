"""Hungarian set-prediction loss.

The model emits n_queries candidates in no particular order and a voxel holds a smaller
number of true compartments, also unordered. Before any error can be computed the two sets
are paired by the Hungarian algorithm (the one-to-one matching of lowest total cost), which
makes the loss permutation-invariant. Matched pairs then get a regression loss on T1, T2
and weight, and every query gets a binary cross-entropy on its existence logit.

Shapes:
    y_pred : (B, n_queries, 4)   [T1, T2, weight, existence_logit] per query
    y_true : (B, max_comp * 3)   [T1, T2, weight] per compartment, flattened
    n_comp : (B,)                number of real compartments; the rest is padding
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianLoss(nn.Module):
    """Set loss with Hungarian matching, configured by a LossConfig.

    `t1_t2_weighting` controls how strongly a compartment's own signal weight w scales its
    T1/T2 error:

      signal_fraction  scaled by w; per-voxel reduction is sum(w * e) / sum(w)
      legacy           scaled by w; reduced by a plain mean over matched pairs
                       (reproduces the very first baseline)
      sqrt             scaled by sqrt(w)
      uniform          no scaling; every compartment counts the same

    Under signal_fraction a 5 % pool receives about fifteen times less gradient than a
    75 % pool. The loss_uniform arm exists to separate that effect from the information
    content of the protocol.
    """

    def __init__(self, cfg):
        super().__init__()
        self.t1_w = cfg.t1_weight
        self.t2_w = cfg.t2_weight
        self.wt_w = cfg.w_weight
        self.ex_w = cfg.exist_weight
        self.t1_t2_weighting = cfg.t1_t2_weighting
        if self.t1_t2_weighting not in {"legacy", "signal_fraction", "uniform", "sqrt"}:
            raise ValueError(
                "t1_t2_weighting must be legacy|signal_fraction|uniform|sqrt; "
                f"got {self.t1_t2_weighting!r}"
            )

    def forward(self, y_pred, y_true, n_comp):
        device = y_pred.device
        B, n_queries, _ = y_pred.shape
        n_reg = y_pred.shape[-1] - 1                       # three regression targets
        y_true = y_true.reshape(B, y_true.shape[1] // n_reg, n_reg)   # (B, max_comp, 3)
        max_comp = y_true.shape[1]

        # 1. Cost of assigning every query to every true compartment, as a (B, Q, C) table.
        p_t1, p_t2, p_wt = y_pred[:, :, 0:1], y_pred[:, :, 1:2], y_pred[:, :, 2:3]
        t_t1 = y_true[:, :, 0].unsqueeze(1)
        t_t2 = y_true[:, :, 1].unsqueeze(1)
        t_wt = y_true[:, :, 2].unsqueeze(1)

        t1_sq = (p_t1 - t_t1) ** 2
        t2_sq = (p_t2 - t_t2) ** 2
        wt_sq = (p_wt - t_wt) ** 2

        cost = self.t1_w * t1_sq + self.t2_w * t2_sq
        # The same scaling is applied in step 3, otherwise the matching would optimise one
        # objective and the loss report another.
        if self.t1_t2_weighting == "sqrt":
            cost = cost * torch.sqrt(t_wt.clamp(min=0.0))
        elif self.t1_t2_weighting != "uniform":
            cost = cost * t_wt
        cost = cost + self.wt_w * wt_sq
        # Confident queries are cheaper to assign, so a true compartment tends to be picked
        # up by a query that already believes it exists.
        exist_prob = torch.sigmoid(y_pred[:, :, 3])
        cost = cost + self.ex_w * (1.0 - exist_prob).unsqueeze(2).expand(-1, -1, max_comp)

        # 2. Solve the assignment per voxel. scipy needs numpy, so the cost table makes a
        # round trip to the CPU. About 5 microseconds per voxel at Q=10, far below the
        # transformer's own cost.
        cost_np = cost.detach().cpu().numpy()
        pred_idx, true_idx, batch_idx = [], [], []
        for b in range(B):
            nc = int(n_comp[b].item() if torch.is_tensor(n_comp[b]) else n_comp[b])
            # Only the first nc columns are real; the rest is zero padding.
            rows, cols = linear_sum_assignment(cost_np[b, :, :nc])
            pred_idx.extend(rows)
            true_idx.extend(cols)
            batch_idx.extend([b] * len(rows))

        p = torch.tensor(pred_idx, device=device, dtype=torch.long)
        t = torch.tensor(true_idx, device=device, dtype=torch.long)
        bidx = torch.tensor(batch_idx, device=device, dtype=torch.long)

        # 3. Regression loss on the matched pairs.
        matched_w = y_true[bidx, t, 2]
        if self.t1_t2_weighting == "uniform":
            fraction = torch.ones_like(matched_w)
        elif self.t1_t2_weighting == "sqrt":
            fraction = torch.sqrt(matched_w.clamp(min=0.0))
        else:
            fraction = matched_w
        weighted_t1 = t1_sq[bidx, p, t] * fraction * self.t1_w
        weighted_t2 = t2_sq[bidx, p, t] * fraction * self.t2_w
        weighted_wt = wt_sq[bidx, p, t] * self.wt_w

        # 4. Existence classification over all queries. Matched queries are the positives.
        # With one to three compartments against ten queries they are heavily outnumbered,
        # so they are up-weighted; without that the head predicts "empty" everywhere.
        exist_tgt = torch.zeros(B, n_queries, device=device)
        exist_tgt[bidx, p] = 1.0
        pos = exist_tgt.sum(dim=-1)
        pos_weight = torch.clamp((n_queries - pos) / pos.clamp(min=1.0), min=0.5, max=10.0)
        cls_per = F.binary_cross_entropy_with_logits(
            y_pred[:, :, 3], exist_tgt, pos_weight=pos_weight.unsqueeze(1), reduction="none"
        ).mean(dim=-1)                                    # one value per voxel

        # 5. Reduce the matched-pair errors to one number per voxel.
        def _per_voxel_mean(vals):
            s = torch.zeros(B, device=device).index_add_(0, bidx, vals)
            c = torch.zeros(B, device=device).index_add_(0, bidx, torch.ones_like(vals))
            return s / c.clamp(min=1.0)

        def _per_voxel_fraction_mean(vals):
            """sum(w * e) / sum(w) per voxel."""
            s = torch.zeros(B, device=device).index_add_(0, bidx, vals)
            w = torch.zeros(B, device=device).index_add_(0, bidx, matched_w)
            return s / w.clamp(min=1e-12)

        if self.t1_t2_weighting == "signal_fraction":
            bt1 = _per_voxel_fraction_mean(weighted_t1)
            bt2 = _per_voxel_fraction_mean(weighted_t2)
        else:
            bt1 = _per_voxel_mean(weighted_t1)
            bt2 = _per_voxel_mean(weighted_t2)
        bwt = _per_voxel_mean(weighted_wt)

        # The components are returned with the total so the log can show where the error sits.
        per_voxel = bt1 + bt2 + bwt + self.ex_w * cls_per
        loss = per_voxel.mean()
        return loss, bt1.mean(), bt2.mean(), bwt.mean(), self.ex_w * cls_per.mean()
