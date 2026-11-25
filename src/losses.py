"""Target encoding (HR, ln RMSSD) and the multi-task loss."""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from configs.config import REG, DISORDERS, STAGES, IGNORE


class Targets:
    """HR is z-scored; RMSSD is log-transformed (ln RMSSD, standard in HRV research: removes the heavy right
       tail) and then z-scored. Statistics come from the TRAINING subjects only."""
    def __init__(self, idx, train_rows):
        self.raw = idx[REG].to_numpy(np.float64)
        t = self._fwd(self.raw)
        self.mu, self.sd = t[train_rows].mean(0), t[train_rows].std(0)
        self.raw_sd = self.raw[train_rows].std(0)                   # for the validation score in original units
    @staticmethod
    def _fwd(raw): return np.stack([raw[:, 0], np.log(np.maximum(raw[:, 1], 1.0))], 1)
    def encode(self): return ((self._fwd(self.raw) - self.mu) / self.sd).astype(np.float32)
    def decode(self, hr_z, rm_z):
        return hr_z * self.sd[0] + self.mu[0], np.exp(rm_z * self.sd[1] + self.mu[1])


class MultiTaskLoss:
    def __init__(self, idx_train, device, w_dis=0.5, w_stg=0.3, smoothing=0.1):
        subj = idx_train.drop_duplicates("record")
        n_dis = np.array([(subj.dis == i).sum() for i in range(len(DISORDERS))], float)      # subjects per class
        w = 1.0 / np.maximum(n_dis, 1); w = w / w.sum() * len(w)
        n_stg = np.array([(idx_train.stg == i).sum() for i in range(len(STAGES))], float)    # windows per stage
        ws = 1.0 / np.sqrt(np.maximum(n_stg, 1)); ws = ws / ws.sum() * len(ws)
        self.ce_dis = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device),
                                          ignore_index=IGNORE, label_smoothing=smoothing)
        self.ce_stg = nn.CrossEntropyLoss(weight=torch.tensor(ws, dtype=torch.float32, device=device),
                                          label_smoothing=smoothing)
        self.w_dis, self.w_stg = w_dis, w_stg
    def __call__(self, out, y):
        hub = lambda a, b: F.smooth_l1_loss(a.float(), b, beta=1.0)                         # Huber: robust to outliers
        parts = {"hr": hub(out["hr"], y["hr"]), "rmssd": hub(out["rmssd"], y["rmssd"])}
        if (y["dis"] != IGNORE).any(): parts["dis"] = self.ce_dis(out["dis"].float(), y["dis"])
        else: parts["dis"] = torch.zeros((), device=y["hr"].device)
        total = parts["hr"] + parts["rmssd"] + self.w_dis * parts["dis"]
        if "stg" in out:
            parts["stg"] = self.ce_stg(out["stg"].float(), y["stg"]); total = total + self.w_stg * parts["stg"]
        return total, parts
