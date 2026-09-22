"""Does synthesising patients improve the *external* validation?

Usage:
    python 03_omics_synthesis/run_experiment.py --demo
    python 03_omics_synthesis/run_experiment.py --data path/to/mtbls1.csv --methods none smote

The experiment is deliberately small: same features, same classifier, same threshold rule; the
only thing that changes is whether the training set was augmented, and by what.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from augment import augment  # noqa: E402
from baselines import run_one  # noqa: E402
from data import load_mtbls1, split, synthetic_cohort  # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="use the synthetic two-batch cohort")
    parser.add_argument("--data", default=None, help="path to a prepared MTBLS1 csv")
    parser.add_argument("--methods", nargs="+", default=["none", "smote", "gaussian_copula"])
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="synthesise until each class reaches ratio x the majority size")
    parser.add_argument("--models", nargs="+", default=["logreg", "gbdt"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-shift", type=float, default=0.9)
    args = parser.parse_args()

    FIGDIR.mkdir(exist_ok=True)
    if args.data:
        frame = load_mtbls1(args.data)
        source = Path(args.data).name
    else:
        frame = synthetic_cohort(batch_shift=args.batch_shift)
        source = "synthetic demo cohort"
    print(f"source: {source}; {len(frame)} samples, "
          f"{(frame.batch != 'internal').sum()} of them in the external batch")

    records, first_batch = [], None
    for seed in args.seeds:
        tr, te, ex, cols = split(frame, seed=seed)
        x_tr_all, y_tr = tr[cols].to_numpy(float), tr["label"].to_numpy()
        x_te, y_te = te[cols].to_numpy(float), te["label"].to_numpy()
        x_ex, y_ex = ex[cols].to_numpy(float), ex["label"].to_numpy()
        for method in args.methods:
            rng = np.random.default_rng(seed)
            x_aug, y_aug = augment(method, x_tr_all, y_tr, args.ratio, rng=rng)
            for model_kind in args.models:
                rec = run_one(method, x_aug, y_aug, x_te, y_te, x_ex, y_ex,
                              model_kind=model_kind, seed=seed)
                rec["synthetic_n"] = len(y_aug) - len(y_tr)
                records.append(rec)
        if first_batch is None:
            first_batch = (x_tr_all, y_tr, x_te, y_te, x_ex, y_ex, cols)

    flat = []
    for r in records:
        flat.append({"method": r["method"], "model": r["model"], "seed": r["seed"],
                     "internal_auc": r["internal"]["auc"],
                     "internal_sens": r["internal"]["sensitivity"],
                     "internal_spec": r["internal"]["specificity"],
                     "external_auc": r["external"]["auc"],
                     "external_sens": r["external"]["sensitivity"],
                     "external_spec": r["external"]["specificity"]})
    table = pd.DataFrame(flat)
    summary = table.groupby(["method", "model"])[
        ["internal_auc", "internal_sens", "internal_spec",
         "external_auc", "external_sens", "external_spec"]].agg(["mean", "std"])

    print("\nmean over seeds (sd in brackets)")
    print(f"{'method':<17}{'model':<8}{'internal AUC':>16}{'internal sens':>16}"
          f"{'external AUC':>16}{'external sens':>16}{'external spec':>16}")
    for (method, model_kind), row in summary.iterrows():
        cells = []
        for key in ["internal_auc", "internal_sens", "internal_spec",
                    "external_auc", "external_sens", "external_spec"]:
            cells.append(f"{row[(key, 'mean')]:.3f} ({row[(key, 'std')]:.3f})")
        # keep the table to the six columns that matter
        print(f"{method:<17}{model_kind:<8}" + "".join(f"{c:>16}" for c in
              [cells[0], cells[1], cells[3], cells[4], cells[5]]))

    table.to_csv(FIGDIR.parent / "outputs_omics_augmentation.csv", index=False)

    # ---- figure: internal vs external, per method ---------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    methods = args.methods
    width = 0.35
    pos = np.arange(len(methods))
    for ax, (metric, title) in zip(axes, [("internal_auc", "internal test AUC"),
                                          ("external_auc", "external validation AUC"),
                                          ("external_spec", "external specificity")]):
        for offset, model_kind, colour in zip([-width / 2, width / 2], args.models,
                                              ["#2b6cb0", "#2f855a"]):
            means = [table[(table.method == m) & (table.model == model_kind)][metric].mean()
                     for m in methods]
            sds = [table[(table.method == m) & (table.model == model_kind)][metric].std()
                   for m in methods]
            ax.bar(pos + offset, means, width, yerr=sds, capsize=3, color=colour,
                   label=model_kind)
        ax.set_xticks(pos)
        ax.set_xticklabels(methods, rotation=12)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Augmentation and the internal/external gap (mean +- sd over seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "03_omics_augmentation.png", dpi=150)
    plt.close(fig)

    # ---- figure: where the synthetic samples land ---------------------------
    x_tr_all, y_tr, x_te, y_te, x_ex, y_ex, cols = first_batch
    mean = x_tr_all.mean(axis=0)
    scale = x_tr_all.std(axis=0) + 1e-9
    z_tr, z_te, z_ex = (x_tr_all - mean) / scale, (x_te - mean) / scale, (x_ex - mean) / scale
    u, s, vt = np.linalg.svd(z_tr - z_tr.mean(0), full_matrices=False)
    project = lambda z: (z - z_tr.mean(0)) @ vt[:2].T

    fig, axes = plt.subplots(1, len(args.methods), figsize=(4.6 * len(args.methods), 4.3),
                             squeeze=False)
    for ax, method in zip(axes[0], args.methods):
        rng = np.random.default_rng(0)
        x_aug, y_aug = augment(method, x_tr_all, y_tr, args.ratio, rng=rng)
        n_new = len(y_aug) - len(y_tr)
        ax.scatter(*project(z_ex).T, s=10, alpha=0.35, color="#718096", label="external batch")
        ax.scatter(*project(z_tr).T, s=14, alpha=0.8, c=y_tr, cmap="coolwarm", marker="o",
                   label="training batch")
        if n_new:
            z_syn = (x_aug[len(y_tr):] - mean) / scale
            ax.scatter(*project(z_syn).T, s=14, alpha=0.8, marker="x", color="k",
                       label=f"{n_new} synthetic")
        ax.set_title(f"method: {method}")
        ax.set_xlabel("PC1")
    axes[0][0].set_ylabel("PC2")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Principal components: the external batch is shifted; where does the generator "
                 "put its samples?", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "03_omics_pca.png", dpi=150)
    plt.close(fig)

    print(f"\nfigures and CSV written to {FIGDIR} and {FIGDIR.parent}")


if __name__ == "__main__":
    main()
