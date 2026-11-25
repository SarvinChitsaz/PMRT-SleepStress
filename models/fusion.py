"""Cross-Modal Fusion block (EEG x EMG features at one stage)."""
import torch, torch.nn as nn


class CrossModalFusion(nn.Module):
    def __init__(self, c_in, d=128, r=16):
        super().__init__()
        self.pa = nn.Sequential(nn.Conv1d(c_in, d, 1, bias=False), nn.BatchNorm1d(d))
        self.pb = nn.Sequential(nn.Conv1d(c_in, d, 1, bias=False), nn.BatchNorm1d(d))
        self.cat = nn.Conv1d(2 * d, d, 1)
        self.gate = nn.Sequential(nn.Flatten(), nn.Linear(2 * d, 2 * d // r), nn.ReLU(inplace=True), nn.Linear(2 * d // r, 3 * d))
        self.s = nn.Conv1d(d, 1, 1); self.off = nn.Conv1d(d, d, 3, 1, 1)
        self.enh = nn.Sequential(nn.Conv1d(d, d, 3, 1, 1, groups=d, bias=False), nn.BatchNorm1d(d), nn.ReLU(inplace=True))
        self.d = d
    def forward(self, fa, fb):
        a, b = self.pa(fa), self.pb(fb); c = torch.cat([a, b], 1)
        w = torch.softmax(self.gate(c.mean(-1, keepdim=True)).view(-1, 3, self.d), dim=1).unsqueeze(-1)
        f = w[:, 0] * (a + b) + w[:, 1] * (a * b) + w[:, 2] * self.cat(c)
        f = self.off(torch.sigmoid(self.s(f)) * f) + f
        return self.enh(f)
