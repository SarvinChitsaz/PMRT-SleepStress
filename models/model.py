"""PMRT with 60-s temporal context: two branches, cross-modal fusion, FPN, multi-scale predictors."""
import torch, torch.nn as nn, torch.nn.functional as F

from configs.config import DISORDERS, STAGES
from models.phy_se_resnet import PhySEResNet
from models.fusion import CrossModalFusion


class MLPHead(nn.Module):                        # [GAP(FP_l), context] -> FC -> ELU -> Dropout -> FC
    def __init__(self, d, ctx, out, hidden=64, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + ctx, hidden), nn.ELU(), nn.Dropout(p), nn.Linear(hidden, out))
    def forward(self, f, c): return self.net(torch.cat([f.mean(-1), c], 1))

class MultiScalePredictor(nn.Module):            # y = sum_l softmax(alpha)_l * h_l(FP_l, context)   (Eq. 9-10)
    def __init__(self, d, ctx, out, levels=3, p=0.3):
        super().__init__()
        self.heads = nn.ModuleList([MLPHead(d, ctx, out, p=p) for _ in range(levels)])
        self.alpha = nn.Parameter(torch.zeros(levels))
    def forward(self, feats, c):
        w = torch.softmax(self.alpha, 0)
        return sum(w[i] * h(f, c) for i, (h, f) in enumerate(zip(self.heads, feats)))

class TemporalContext(nn.Module):
    """Shares information between the windows of a sequence (bidirectional, padding masked)."""
    def __init__(self, d=128, L=12, layers=2, heads=4, p=0.1):
        super().__init__()
        self.proj = nn.Linear(3 * d, d); self.pos = nn.Parameter(torch.zeros(1, L, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, p, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
    def forward(self, v, mask):                  # v: (B, L, 3d), mask: (B, L) True = real window
        h = self.proj(v) + self.pos[:, :v.shape[1]]
        return self.enc(h, src_key_padding_mask=~mask)

class PMRTTemporal(nn.Module):
    def __init__(self, d=128, L=12, deriv_kernel=15, r=16, p=0.3, feat_drop=0.1, stage_head=True, input_mode="both"):
        super().__init__()
        assert input_mode in ("both", "eeg", "emg")
        self.input_mode = input_mode                                           # ablation: "eeg" / "emg" = other input set to zero
        self.branch_eeg = PhySEResNet(deriv_kernel=deriv_kernel, r=r)          # network 1 (Fig. 4)
        self.branch_emg = PhySEResNet(deriv_kernel=deriv_kernel, r=r)          # network 2 (Fig. 4)
        self.fusion = nn.ModuleList([CrossModalFusion(c, d, r) for c in (128, 256, 512)])   # Fig. 5
        self.fdrop = nn.Dropout1d(feat_drop)
        self.context = TemporalContext(d, L)
        outs = {"hr": 1, "rmssd": 1, "dis": len(DISORDERS)}
        if stage_head: outs["stg"] = len(STAGES)
        self.pred = nn.ModuleDict({k: MultiScalePredictor(d, d, o, p=p) for k, o in outs.items()})
    def windows(self, x):                        # one 5-s window -> FP1, FP2, FP3 (exactly the paper's pipeline)
        fa = self.branch_eeg(x[:, :1])[1:]; fb = self.branch_emg(x[:, 1:])[1:]
        m = [fu(a, b) for fu, a, b in zip(self.fusion, fa, fb)]
        p3 = m[2]
        p2 = m[1] + F.interpolate(p3, size=m[1].shape[-1], mode="nearest")
        p1 = m[0] + F.interpolate(p2, size=m[0].shape[-1], mode="nearest")
        return [self.fdrop(q) for q in (p1, p2, p3)]
    def forward(self, x, mask):                  # x: (N_valid, 2, 625) in row-major order of mask (B, L)
        if self.input_mode == "eeg": x = torch.cat([x[:, :1], torch.zeros_like(x[:, 1:])], 1)
        if self.input_mode == "emg": x = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:]], 1)
        fp = self.windows(x)
        v = torch.cat([q.mean(-1) for q in fp], 1)                             # (N, 3d)
        V = v.new_zeros(*mask.shape, v.shape[1]); V[mask] = v
        c = self.context(V, mask)[mask]                                        # (N, d) context of each window
        out = {k: p(fp, c) for k, p in self.pred.items()}
        out["hr"], out["rmssd"] = out["hr"][:, 0], out["rmssd"][:, 0]
        return out
