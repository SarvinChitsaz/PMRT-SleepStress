"""Prediction and validation metrics."""
import numpy as np, pandas as pd
import torch

from configs.config import DISORDERS, IGNORE


@torch.no_grad()
def predict(model, loader):
    model.eval(); res, J = {}, []
    for x, mask, _, j in loader:
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
            out = model(x, mask)
        for k, v in out.items():
            v = v.float()
            res.setdefault(k, []).append((torch.softmax(v, -1) if v.ndim == 2 else v).cpu().numpy())
        J.append(j)
    J = np.concatenate(J); o = np.argsort(J, kind="stable")                    # back to row order
    return J[o], {k: np.concatenate(v)[o] for k, v in res.items()}

def reg_metrics(y, p):
    e = p - y
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))), "ME": float(np.mean(e)),
            "SD": float(np.std(e)), "r": float(np.corrcoef(y, p)[0, 1]),
            "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}

def val_scores(p, j, tg, dis, stg, rec):
    """stress score  = MAE(HR)/SD(HR) + MAE(RMSSD)/SD(RMSSD) in original units (lower = better)
       disorder loss = cross-entropy of the night-averaged probabilities, per subject (lower = better)"""
    hr, rm = tg.decode(p["hr"], p["rmssd"])
    hr_mae = float(np.mean(np.abs(hr - tg.raw[j, 0]))); rm_mae = float(np.mean(np.abs(rm - tg.raw[j, 1])))
    s = {"stress_score": hr_mae / tg.raw_sd[0] + rm_mae / tg.raw_sd[1], "hr_MAE": hr_mae, "rmssd_MAE": rm_mae}
    d = pd.DataFrame(p["dis"], columns=DISORDERS); d["rec"] = rec[j]; d["y"] = dis[j]
    d = d[d.y != IGNORE].groupby("rec").agg({**{c: "mean" for c in DISORDERS}, "y": "first"})
    if len(d):
        pr = d[DISORDERS].to_numpy()
        s["dis_subject_loss"] = float(-np.mean(np.log(pr[np.arange(len(d)), d.y.to_numpy()] + 1e-8)))
        s["dis_subject_acc"] = float(np.mean(pr.argmax(1) == d.y.to_numpy()))
    else:
        s["dis_subject_loss"], s["dis_subject_acc"] = float("nan"), float("nan")
    if "stg" in p: s["stage_acc"] = float(np.mean(p["stg"].argmax(1) == stg[j]))
    return s
