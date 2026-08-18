#!/usr/bin/env python3
"""
Summarize shape features (L_xy, L_z, Rg) of transmembrane proteins whose
residue length N_residues lies in [375, 475], per category, and write the
result to a txt report.

Input : data/process/tm_plddt70.h5  (read-only)
Output: results/shape_feature_statistic/stats_375_475.txt
Run from the repo root (relative paths).

All features are per-protein attrs already stored in the H5 (Å):
  L_xy = sqrt(Gxx + Gyy),  L_z = sqrt(Gzz),  Rg = sqrt(tr(G))
(CA-only, equal weights; the identity Rg² = L_xy² + L_z² holds per protein.)
Note: on means the identity only holds approximately, because the mean of
squares differs from the square of the mean. That is expected, not a bug.

Dependencies: h5py, numpy
"""

import sys
from pathlib import Path

import h5py
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
H5_PATH   = "data/process/tm_plddt70.h5"
OUT_DIR   = "results/shape_feature_statistic"
OUT_FNAME = "stats_375_475.txt"

N_RES_MIN  = 375
N_RES_MAX  = 475

CATEGORIES = ["GPCR", "large", "medium", "single_pass"]
FEATURES   = ["L_xy", "L_z", "Rg"]

# Display labels for category rows (H5 group names are used as lookup keys)
CATEGORY_LABELS = {"GPCR": "7TM(GPCR)"}

# ── Column layout (aligned rows) ─────────────────────────────────────────────
NAME_W  = 12
COUNT_W = 7
MEAN_W  = 11

HEADER = (f"{'category':<{NAME_W}s}{'count':>{COUNT_W}s}"
          + "".join(f"{f'mean_{f}':>{MEAN_W}s}" for f in FEATURES))


def fmt_row(name, count, means):
    """Format one aligned row: name / count / mean of each feature."""
    cells = [f"{name:<{NAME_W}s}", f"{count:>{COUNT_W}d}"]
    cells += [f"{means[f]:>{MEAN_W}.4f}" for f in FEATURES]
    return "".join(cells)


def collect_stats(h5_path, categories, n_min, n_max):
    """Read per-protein shape features from the H5 and aggregate.

    Iterates each category group, skips the ``protein_ids`` dataset, and
    keeps proteins with n_min <= N_residues <= n_max. Proteins missing any
    of the feature attrs inside the window are tallied and skipped.

    Parameters
    ----------
    h5_path : str
        Path to tm_plddt70.h5 (opened read-only).
    categories : list of str
        Category group names to process.
    n_min, n_max : int
        Inclusive N_residues window.

    Returns
    -------
    stats : dict
        {category: {"count": n, "means": {feature: float}}} plus a "total"
        entry combining all categories (weighted by protein count).
    n_incomplete : int
        Windowed proteins missing one of the features (skipped).
    """
    stats = {}
    n_incomplete = 0

    with h5py.File(h5_path, "r") as h5f:
        for cat in categories:
            if cat not in h5f:
                print(f"  ⚠ category '{cat}' not found, skipping",
                      file=sys.stderr)
                continue

            grp = h5f[cat]
            vals = {f: [] for f in FEATURES}

            for uid in grp.keys():
                if uid == "protein_ids":          # skip the ID dataset
                    continue
                prot = grp[uid]
                if not isinstance(prot, h5py.Group):
                    continue

                n_res = int(prot.attrs.get("N_residues", -1))
                if n_res < n_min or n_res > n_max:
                    continue

                if any(f not in prot.attrs for f in FEATURES):
                    n_incomplete += 1
                    continue

                for f in FEATURES:
                    vals[f].append(float(prot.attrs[f]))

            stats[cat] = {"values": vals}

    # ── Aggregate: count + means per category, plus a combined total ─────
    total_vals = {f: [] for f in FEATURES}
    for cat, entry in stats.items():
        vals = entry["values"]
        entry["count"] = len(vals["L_xy"])
        entry["means"] = {
            f: float(np.mean(vals[f])) if vals[f] else float("nan")
            for f in FEATURES
        }
        for f in FEATURES:
            total_vals[f].extend(vals[f])

    stats["total"] = {
        "count": len(total_vals["L_xy"]),
        "means": {
            f: float(np.mean(total_vals[f])) if total_vals[f] else float("nan")
            for f in FEATURES
        },
    }

    return stats, n_incomplete


def main():
    print(f"Reading: {H5_PATH}")
    stats, n_incomplete = collect_stats(H5_PATH, CATEGORIES,
                                        N_RES_MIN, N_RES_MAX)

    lines = [
        f"形状特征统计（N_residues ∈ [{N_RES_MIN}, {N_RES_MAX}]）",
        f"数据: {H5_PATH}      单位: Å",
        "",
        "── 1. 四个类别 ──",
        HEADER,
    ]
    for cat in CATEGORIES:
        if cat in stats:
            lines.append(fmt_row(CATEGORY_LABELS.get(cat, cat),
                                 stats[cat]["count"],
                                 stats[cat]["means"]))
    lines.append(fmt_row("total", stats["total"]["count"],
                         stats["total"]["means"]))
    lines.append("")

    # ── Section 2: GPCR alone ─────────────────────────────────────────
    # User asked for the GPCR category as its own section; identical to the
    # GPCR row above — do not "deduplicate".
    lines.append("── 2. GPCR 类别 ──")
    lines.append(HEADER)
    if "GPCR" in stats:
        lines.append(fmt_row(CATEGORY_LABELS.get("GPCR", "GPCR"),
                             stats["GPCR"]["count"],
                             stats["GPCR"]["means"]))

    if n_incomplete:
        lines.append("")
        lines.append(f"注: {n_incomplete} 个窗口内蛋白因缺少 "
                     f"L_xy/L_z/Rg attr 被跳过。")

    report = "\n".join(lines) + "\n"

    out_path = Path(OUT_DIR) / OUT_FNAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
