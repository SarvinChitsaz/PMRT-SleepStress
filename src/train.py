"""Training of one fold (resumable) and the 5-fold cross-validation loop."""
import json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from pathlib import Path

from configs.config import REG, DISORDERS, STAGES, VERSION
from data.dataset import (load_data, make_folds, fold_rows, SequenceBatches, augment)
from models.model import PMRTTemporal
from src.losses import Targets, MultiTaskLoss
from src.eval import predict, val_scores
from src.utils import EMA


def train_fold(model, tr, va, loss_fn, tg, dis, stg, rec, run_dir, epochs=60, lr=5e-4, wd=0.05,
               patience=12, min_epochs=15, use_aug=True, ema_decay=0.999):
    """Same protocol as mt2: best_stress.pt (validation stress score) and best_dis.pt (subject-level
       validation disorder loss); early stopping after min_epochs when neither improved for `patience` epochs."""
    run = Path(run_dir); run.mkdir(parents=True, exist_ok=True)
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(tr),
                                                pct_start=max(0.1, 3 / (epochs * len(tr))))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    ema = EMA(model, ema_decay)
    state = {"epoch": 0, "best_stress": float("inf"), "best_dis": float("inf"), "bad": 0, "history": []}
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.m.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"]); state = ck["state"]
        print(f"  resumed after epoch {state['epoch']}")
    while state["epoch"] < epochs and not (state["epoch"] >= min_epochs and state["bad"] >= patience):
        ep = state["epoch"] + 1; t0 = time.time(); model.train(); tot, nb = 0.0, 0
        for step, (x, mask, y, _) in enumerate(tr):
            if step >= len(tr): break                                          # keep OneCycle step count exact
            if use_aug: x = augment(x)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                out = model(x, mask)
            loss, _ = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
            tot += loss.item(); nb += 1
        j, p = predict(ema.m, va)
        s = val_scores(p, j, tg, dis, stg, rec)
        rec_h = {"epoch": ep, "train_loss": tot / nb, **{f"val_{k}": v for k, v in s.items()},
                 "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0}
        state["history"].append(rec_h); state["epoch"] = ep
        flag, improved = "", False
        if s["stress_score"] < state["best_stress"]:
            state["best_stress"] = s["stress_score"]; improved = True; flag += " *stress*"
            torch.save({"model": ema.m.state_dict(), "epoch": ep}, run / "best_stress.pt")
        if s["dis_subject_loss"] < state["best_dis"]:
            state["best_dis"] = s["dis_subject_loss"]; improved = True; flag += " *dis*"
            torch.save({"model": ema.m.state_dict(), "epoch": ep}, run / "best_dis.pt")
        state["bad"] = 0 if improved else state["bad"] + 1
        torch.save({"model": model.state_dict(), "ema": ema.m.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "state": state}, run / "last.pt")
        pd.DataFrame(state["history"]).to_csv(run / "history.csv", index=False)
        print(f"  ep {ep:3d} | train {rec_h['train_loss']:.3f} | stress score {s['stress_score']:.3f} | HR MAE {s['hr_MAE']:.2f} | "
              f"RMSSD MAE {s['rmssd_MAE']:.1f} | dis subj acc {s['dis_subject_acc']:.2f} loss {s['dis_subject_loss']:.2f}" +
              (f" | stage acc {s['stage_acc']:.2f}" if "stage_acc" in s else "") + f" | {rec_h['sec']:.0f}s{flag}", flush=True)
    h = pd.DataFrame(state["history"])
    print(f"  finished after {state['epoch']} epochs | best stress epoch {int(h.loc[h.val_stress_score.idxmin(), 'epoch'])} "
          f"| best disorder epoch {int(h.loc[h.val_dis_subject_loss.idxmin(), 'epoch'])}")


def run_cv(cache_dir, run_dir, folds=None, n_folds=None, seq_len=None, epochs=None, patience=None,
           min_epochs=None, batch_seq=None, lr=None, wd=None, per_subject=None, use_aug=None,
           use_stage_head=None, ema_decay=None, seed=None, device=None):
    """Subject-wise cross-validation. Every fold: train, pick the two checkpoints, write fold{k}/test_pred.csv.
       Finished folds are skipped, so the run can be resumed."""
    import configs.config as C
    n_folds = n_folds or C.N_FOLDS; seq_len = seq_len or C.SEQ_LEN; epochs = epochs or C.EPOCHS
    patience = patience or C.PATIENCE; min_epochs = min_epochs or C.MIN_EPOCHS
    batch_seq = batch_seq or C.BATCH_SEQ; lr = lr or C.LR; wd = wd or C.WD
    per_subject = per_subject or C.PER_SUBJECT
    use_aug = C.USE_AUG if use_aug is None else use_aug
    use_stage_head = C.USE_STAGE_HEAD if use_stage_head is None else use_stage_head
    ema_decay = ema_decay or C.EMA_DECAY; seed = C.SEED if seed is None else seed
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "- WARNING: no GPU")

    X, idx = load_data(cache_dir)
    REC = idx.record.to_numpy()
    cache = Path(cache_dir)
    fold_file = cache / f"folds_{n_folds}.json"
    if not fold_file.exists():
        fold_file.write_text(json.dumps(make_folds(idx, n_folds, seed), indent=1))
    FOLD = json.loads(fold_file.read_text())

    RUN = Path(run_dir); RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "config.json").write_text(json.dumps(
        dict(N_FOLDS=n_folds, SEQ_LEN=seq_len, EPOCHS=epochs, PATIENCE=patience, MIN_EPOCHS=min_epochs,
             BATCH_SEQ=batch_seq, LR=lr, WD=wd, PER_SUBJECT=per_subject, USE_AUG=use_aug,
             USE_STAGE_HEAD=use_stage_head, EMA_DECAY=ema_decay, SEED=seed, version=VERSION), indent=1))
    DIS, STG = idx.dis.to_numpy(np.int64), idx.stg.to_numpy(np.int64)

    for k in (range(n_folds) if folds is None else folds):
        fdir = RUN / f"fold{k}"
        if (fdir / "test_pred.csv").exists():
            print(f"fold {k}: already finished - skipped"); continue
        torch.manual_seed(seed + k); np.random.seed(seed + k)
        rows = fold_rows(idx, FOLD, k, n_folds)
        tg = Targets(idx, rows["train"])
        yz = tg.encode()
        Y = {"hr": yz[:, 0], "rmssd": yz[:, 1], "dis": DIS, "stg": STG}
        tr = SequenceBatches(X, Y, idx, rows["train"], batch_seq, dev, seq_len, per_subject, train=True)
        va = SequenceBatches(X, Y, idx, rows["val"], C.EVAL_BATCH_SEQ, dev, seq_len, train=False)
        te = SequenceBatches(X, Y, idx, rows["test"], C.EVAL_BATCH_SEQ, dev, seq_len, train=False)
        print(f"\n===== fold {k} | train {len(np.unique(REC[rows['train']]))} subj | val {len(np.unique(REC[rows['val']]))} "
              f"| test {len(np.unique(REC[rows['test']]))} =====")
        loss_fn = MultiTaskLoss(idx.iloc[rows["train"]], dev)
        model = PMRTTemporal(L=seq_len, stage_head=use_stage_head).to(dev)
        train_fold(model, tr, va, loss_fn, tg, DIS, STG, REC, fdir, epochs=epochs, lr=lr, wd=wd,
                   patience=patience, min_epochs=min_epochs, use_aug=use_aug, ema_decay=ema_decay)

        model.load_state_dict(torch.load(fdir / "best_stress.pt", map_location=dev, weights_only=False)["model"])
        j, p = predict(model, te)
        model.load_state_dict(torch.load(fdir / "best_dis.pt", map_location=dev, weights_only=False)["model"])
        j2, p2 = predict(model, te)
        assert (j == j2).all() and (j == np.sort(rows["test"])).all()
        out = idx.iloc[j][["record", "group", "stage", "clipped", "dis", "stg"] + REG].reset_index(drop=True)
        out["fold"] = k
        out["hr_pred"], out["rmssd_pred"] = tg.decode(p["hr"], p["rmssd"])
        for c, name in enumerate(DISORDERS): out[f"p_{name}"] = p2["dis"][:, c]
        if "stg" in p:
            for c, name in enumerate(STAGES): out[f"p_{name}"] = p["stg"][:, c]
        out["mu_hr"], out["mu_rmssd"] = np.median(tg.raw[rows["train"]], 0)
        out.to_csv(fdir / "test_pred.csv", index=False)
        print(f"fold {k} saved")
    print("\nall folds finished")
