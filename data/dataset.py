"""Data handling: cache building, subject-wise folds, batch iterators, augmentation."""
import json, glob, math
import numpy as np, pandas as pd
import torch
from pathlib import Path

from configs.config import DISORDERS, STAGES, IGNORE


def build_cache(npz_dir, cache_dir, tmp_dir="/content/_cache_tmp"):
    """All per-record npz files -> one x.npy + index.csv (one-time checkpoint, built locally then copied)."""
    import shutil
    final = Path(cache_dir)
    if (final / "index.csv").exists():
        return
    cache = Path(tmp_dir); shutil.rmtree(cache, ignore_errors=True); cache.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{npz_dir}/*.npz"))
    metas, n = [], 0
    for f in files:
        with np.load(f) as d:
            metas.append((f, json.loads(str(d["meta"])), len(d["hr"]))); n += len(d["hr"])
    X = np.lib.format.open_memmap(cache / "x.npy", mode="w+", dtype=np.float16, shape=(n, 2, 625))
    rows, pos = [], 0
    for f, m, k in metas:
        with np.load(f) as d:
            X[pos:pos + k, 0] = d["eeg"]; X[pos:pos + k, 1] = d["emg"]
            clip = (d["eeg_clip"].astype(np.float32) > 0.01) | (d["emg_clip"].astype(np.float32) > 0.01)
            rows.append(pd.DataFrame({"record": m["record"], "group": m["group"], "age": m["age"], "sex": m["sex"],
                                      "stage": d["stage"], "t": d["t"], "hr": d["hr"], "rmssd": d["rmssd"],
                                      "clipped": clip}))
        pos += k
        print(f"cached {m['record']:8s} {pos:,}/{n:,}", flush=True)
    X.flush(); del X
    pd.concat(rows, ignore_index=True).to_csv(cache / "index.csv", index=False)
    final.mkdir(parents=True, exist_ok=True)
    shutil.copy(cache / "x.npy", final / "x.npy"); shutil.copy(cache / "index.csv", final / "index.csv")
    shutil.rmtree(cache, ignore_errors=True)

def load_data(cache_dir):
    """Loads everything into RAM and adds integer labels for the classification heads."""
    X = np.load(Path(cache_dir) / "x.npy")
    idx = pd.read_csv(Path(cache_dir) / "index.csv")
    idx["dis"] = idx.group.map({g: i for i, g in enumerate(DISORDERS)}).fillna(IGNORE).astype(int)
    idx["stg"] = idx.stage.map({s: i for i, s in enumerate(STAGES)}).astype(int)
    return X, idx

def make_folds(idx, k=5, seed=42):
    """Subject-wise folds, stratified by diagnostic group: each subject belongs to exactly one fold."""
    rng = np.random.default_rng(seed); fold, start = {}, 0
    for g, recs in sorted(idx.groupby("group").record.unique().items()):
        recs = sorted(recs); rng.shuffle(recs)
        for i, r in enumerate(recs):
            fold[r] = int((start + i) % k)
        start += len(recs)                                          # rotate so small groups do not all land in fold 0
    return fold

def fold_rows(idx, fold, k_test, k=5):
    """test = fold k_test, validation = fold k_test+1, training = remaining 3 folds."""
    f = idx.record.map(fold).to_numpy()
    k_val = (k_test + 1) % k
    return {"train": np.where((f != k_test) & (f != k_val))[0],
            "val": np.where(f == k_val)[0], "test": np.where(f == k_test)[0]}

class SubjectBalancedBatches:
    """Each epoch draws at most `per_subject` random windows from every training subject.
       -> every subject has the same influence, epochs see different windows (less memorisation)."""
    def __init__(self, X, Y, rows, rec, batch, device, per_subject=1500, shuffle=True):
        self.X, self.Y, self.b, self.dev, self.k, self.sh = X, Y, batch, device, per_subject, shuffle
        self.groups = [rows[rec[rows] == r] for r in np.unique(rec[rows])] if shuffle else None
        self.rows = rows
    def _epoch_rows(self):
        if not self.sh: return self.rows
        pick = [g if len(g) <= self.k else np.random.choice(g, self.k, replace=False) for g in self.groups]
        return np.random.permutation(np.concatenate(pick))
    def __len__(self):
        n = sum(min(len(g), self.k) for g in self.groups) if self.sh else len(self.rows)
        return math.ceil(n / self.b)
    def __iter__(self):
        r = self._epoch_rows()
        for i in range(0, len(r), self.b):
            j = np.sort(r[i:i + self.b])
            x = torch.from_numpy(np.asarray(self.X[j], dtype=np.float32)).to(self.dev, non_blocking=True)
            y = {k: torch.from_numpy(v[j]).to(self.dev, non_blocking=True) for k, v in self.Y.items()}
            yield x, y, j

def augment(x):
    """Label-preserving augmentation on GPU (no time stretching: it would change the heart rate)."""
    B = x.shape[0]; dev = x.device
    x = x * torch.empty(B, 2, 1, device=dev).uniform_(0.8, 1.2)                         # amplitude scaling
    x = x * torch.where(torch.rand(B, 2, 1, device=dev) < 0.5, -1.0, 1.0)              # polarity inversion
    x = x + 0.05 * torch.randn_like(x)                                                   # additive noise
    L = x.shape[-1]; m = torch.rand(B, device=dev) < 0.3                                 # time masking (<=0.4 s)
    if m.any():
        s = torch.randint(0, L - 50, (B,), device=dev); w = torch.randint(10, 50, (B,), device=dev)
        t = torch.arange(L, device=dev)[None]
        mask = (t >= s[:, None]) & (t < (s + w)[:, None]) & m[:, None]
        x = x.masked_fill(mask[:, None, :], 0.0)
    drop = torch.rand(B, device=dev) < 0.1                                               # modality dropout
    if drop.any():
        ch = torch.randint(0, 2, (B,), device=dev)
        keep = torch.ones(B, 2, 1, device=dev)
        keep[drop, ch[drop], 0] = 0.0
        x = x * keep
    return x


# ---------------- sequences of consecutive windows ----------------
def make_sequences(idx, rows, L=12, max_gap=60.0, offset=0):
    """Rows -> (S, L) array of consecutive windows of the same subject (-1 = padding).
       A sequence never crosses a gap > max_gap seconds (wake, rejected windows).
       offset shifts the chunk boundaries (random per epoch in training -> different contexts each epoch)."""
    sub = idx.iloc[rows][["record", "t"]].assign(row=rows).sort_values(["record", "t"])
    seqs = []
    for _, g in sub.groupby("record", sort=False):
        r, t = g.row.to_numpy(), g.t.to_numpy()
        breaks = np.where(np.diff(t) > max_gap)[0] + 1
        for seg in np.split(r, breaks):
            o = offset % L if len(seg) > L else 0
            cuts = list(range(o, len(seg), L)) if o else list(range(0, len(seg), L))
            if o: cuts = [0] + cuts
            for a, b in zip(cuts, cuts[1:] + [len(seg)]):
                chunk = seg[a:b]
                if len(chunk): seqs.append(np.pad(chunk, (0, L - len(chunk)), constant_values=-1))
    return np.array(seqs, dtype=np.int64).reshape(-1, L)

class SequenceBatches:
    """Batches of sequences. Training: new random chunk offset every epoch + at most `per_subject` windows
       per subject (subject-balanced). Evaluation: fixed chunks covering every window exactly once."""
    def __init__(self, X, Y, idx, rows, batch_seq, device, L=12, per_subject=1500, train=True):
        self.X, self.Y, self.idx, self.rows, self.b, self.dev, self.L = X, Y, idx, rows, batch_seq, device, L
        self.k, self.train = per_subject // L, train
        self.rec = idx.record.to_numpy()
        self.fixed = None if train else make_sequences(idx, rows, L)
    def _epoch(self):
        if not self.train: return self.fixed
        s = make_sequences(self.idx, self.rows, self.L, offset=np.random.randint(self.L))
        own = self.rec[s[:, 0]]
        keep = np.concatenate([np.random.permutation(np.where(own == r)[0])[:self.k] for r in np.unique(own)])
        return s[np.random.permutation(keep)]
    def __len__(self):
        n = len(self.fixed) if not self.train else len(np.unique(self.rec[self.rows])) * self.k
        return math.ceil(n / self.b)
    def __iter__(self):
        S = self._epoch()
        for i in range(0, len(S), self.b):
            s = S[i:i + self.b]; m = s >= 0
            j = s[m]                                                   # valid row indices, row-major order
            x = torch.from_numpy(np.asarray(self.X[j], dtype=np.float32)).to(self.dev, non_blocking=True)
            y = {k: torch.from_numpy(v[j]).to(self.dev, non_blocking=True) for k, v in self.Y.items()}
            yield x, torch.from_numpy(m).to(self.dev), y, j
