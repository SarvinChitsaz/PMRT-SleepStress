"""Smoke test on small synthetic data (CPU, about a minute).

Checks that build-cache and the 5-fold training loop run end to end and write the expected files.
It does NOT check accuracy; the data are random.

    python tests/smoke_test.py
"""
import json, sys, tempfile
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import build_cache
from src.train import run_cv
from configs.config import DISORDERS, STAGES

GROUPS = ["Healthy"] * 4 + ["NFLE"] * 2 + ["RBD"] * 2 + ["PLM"] * 2 + ["Insomnia"] * 2 + ["Narcolepsy"] * 2 + ["SDB"]


def make_records(out, n=48, seed=0):
    rng = np.random.default_rng(seed); out.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(GROUPS):
        np.savez(out / f"r{i:02d}.npz",
                 eeg=rng.normal(size=(n, 625)).astype(np.float16), emg=rng.normal(size=(n, 625)).astype(np.float16),
                 eeg_clip=np.zeros(n, np.float16), emg_clip=np.zeros(n, np.float16),
                 stage=rng.choice(STAGES, n), t=np.arange(n) * 5.0,
                 hr=rng.normal(65, 5, n), rmssd=rng.uniform(15, 80, n),
                 meta=json.dumps({"record": f"r{i:02d}", "group": g, "age": 40, "sex": "F"}))


def main():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_records(d / "npz")
        build_cache(str(d / "npz"), str(d / "cache"), str(d / "tmp"))
        assert (d / "cache" / "x.npy").exists() and (d / "cache" / "index.csv").exists()
        run_cv(str(d / "cache"), str(d / "run"), folds=[0], epochs=1, min_epochs=1, device="cpu")
        pred = pd.read_csv(d / "run" / "fold0" / "test_pred.csv")
        need = {"record", "group", "hr", "rmssd", "hr_pred", "rmssd_pred", *[f"p_{c}" for c in DISORDERS + STAGES]}
        assert need <= set(pred.columns), need - set(pred.columns)
        assert np.isfinite(pred[["hr_pred", "rmssd_pred"]].to_numpy()).all()
        for f in ("best_stress.pt", "best_dis.pt", "last.pt", "history.csv"):
            assert (d / "run" / "fold0" / f).exists(), f
    print("smoke test passed")


if __name__ == "__main__":
    main()
