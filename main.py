"""PMRT sleep-stress: command-line entry point.

  python main.py build-cache --npz-dir <cap_processed> --cache-dir <cap_dataset_cache>
  python main.py train       --cache-dir <cap_dataset_cache> --run-dir runs/pmrt_mt3_cv5
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-cache", help="per-record .npz files -> x.npy + index.csv")
    b.add_argument("--npz-dir", required=True)
    b.add_argument("--cache-dir", required=True)
    b.add_argument("--tmp-dir", default="/tmp/_cache_tmp")

    t = sub.add_parser("train", help="5-fold subject-wise cross-validation (resumable)")
    t.add_argument("--cache-dir", required=True)
    t.add_argument("--run-dir", required=True)
    t.add_argument("--folds", type=int, nargs="+", default=None, help="subset of folds, e.g. --folds 0 1")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--device", default=None)

    a = ap.parse_args()
    if a.cmd == "build-cache":
        from data.dataset import build_cache
        build_cache(a.npz_dir, a.cache_dir, a.tmp_dir)
    else:
        from src.train import run_cv
        run_cv(a.cache_dir, a.run_dir, folds=a.folds, epochs=a.epochs, seed=a.seed, device=a.device)


if __name__ == "__main__":
    main()
