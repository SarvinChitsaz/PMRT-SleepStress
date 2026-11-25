"""Phy-SE-ResNet branch (one per input: EEG, chin EMG)."""
import torch, torch.nn as nn, torch.nn.functional as F


class StemBlock(nn.Module):
    def __init__(self, c_out=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, c_out, 7, 2, 3, bias=False), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True),
                                 nn.MaxPool1d(3, 2, 1))
    def forward(self, x): return self.net(x)

class PhySEResidualBlock(nn.Module):
    """X_{l+1} = ReLU( A1*F_main + A2*F_deriv + H(X_l) )   (Eq. 1-4)"""
    def __init__(self, c_in, c_out, stride=2, deriv_kernel=15, r=16, lam=1.0):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm1d(c_out))
        self.deriv = nn.Sequential(
            nn.Conv1d(c_in, c_out, deriv_kernel, stride, deriv_kernel // 2, bias=False), nn.BatchNorm1d(c_out), nn.Hardtanh())
        self.lam = nn.Parameter(torch.tensor(float(lam)))
        h = max(c_out // r, 4)
        self.ub, self.ua = nn.Linear(c_out, h), nn.Linear(h, 2 * c_out)
        self.short = (nn.Identity() if (c_in == c_out and stride == 1) else
                      nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out)))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        fm = self.main(x); fd = self.deriv(x) * self.lam
        a = self.ua(F.relu(self.ub(fm.mean(-1))))
        a = torch.softmax(a.view(-1, 2, fm.shape[1]), dim=1).unsqueeze(-1)
        return self.act(a[:, 0] * fm + a[:, 1] * fd + self.short(x))

class PhySEResNet(nn.Module):
    def __init__(self, widths=(64, 128, 256, 512), **kw):
        super().__init__()
        self.stem = StemBlock(widths[0])
        c, st = widths[0], []
        for w in widths: st.append(PhySEResidualBlock(c, w, stride=2, **kw)); c = w
        self.stages = nn.ModuleList(st)
    def forward(self, x):
        x, feats = self.stem(x), []
        for s in self.stages: x = s(x); feats.append(x)
        return feats
