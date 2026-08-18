#!/usr/bin/env python3
"""
plot_corr_scatter.py — 方向分辨余弦相关性 ρ(r) 分组均值曲线。

固定 s=16, K = 1000 ，只出两张归一化距离图：
  - corr_xy_norm_xy_s16.png : xy 方向，距离按 L_xy 归一化
  - corr_z_norm_z_s16.png   : z  方向，距离按 L_z  归一化

每条曲线为残基数分组均值，叠加 ±1 SD 色带。数据直接来自 HDF5 的 bin_r / bin_rho。

"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ── 参数 ──────────────────────────────────────────────────────────────────
H5_PATH    = "data/process/tm_plddt70.h5"
OUT_DIR    = "results/bin_cor_4_length"
S_DEFAULT  = 16
K_DEFAULT  = 1000

# 残基数量分组（区间两端均包含）
N_RES_BINS = [
    ("275 < N ≤ 375", 275, 375),
    ("325 < N ≤ 425", 325, 425),
    ("375 < N ≤ 475", 375, 475),
    ("425 < N ≤ 525", 425, 525),
]

BIN_COLORS  = {"275 < N ≤ 375": "#E63946", "325 < N ≤ 425": "#264653",
               "375 < N ≤ 475": "#F4A261", "425 < N ≤ 525": "#2A9D8F"}
BIN_MARKERS = {"275 < N ≤ 375": "o", "325 < N ≤ 425": "s", "375 < N ≤ 475": "^", "425 < N ≤ 525": "D"}

ALL_CATEGORIES = ["GPCR", "large", "medium", "single_pass"]
C_NULL = "#999999"
MM = 1.0 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 6, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": True, "legend.handlelength": 1.4,
    "axes.spines.top": True, "axes.spines.right": True,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

NORM_BIN_WIDTH = 0.2   # 归一化 bin 宽度

DIR_LABELS = {
    "xy_norm_xy": r"$\phi_{xy}$",
    "z_norm_z":    r"$\phi_{z}$",
}


def get_n_res_bin(N_res):
    """返回蛋白所属的残基数分组标签，不在任何区间则返回 None。"""
    for label, lo, hi in N_RES_BINS:
        if lo < N_res <= hi:
            return label
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  数据提取：归一化距离 r/L（每个蛋白除以其自身的 L_xy 或 L_z）
# ═══════════════════════════════════════════════════════════════════════════
def load_corr_points_normalized(h5_path, s, K, direction, norm_attr="L_xy"):
    """读取每个蛋白的归一化 (r/L, rho)，按 category 分组。

    每个蛋白的 bin_r 除以其自身的 norm_attr（L_xy 或 L_z），
    然后 snap 到公共归一化 bin grid（宽度 NORM_BIN_WIDTH）。
    """
    bin_labels = [label for label, _, _ in N_RES_BINS]
    curve  = {bl: defaultdict(list) for bl in bin_labels}
    counts = {bl: 0 for bl in bin_labels}

    with h5py.File(h5_path, "r") as hf:
        for cat in ALL_CATEGORIES:
            if cat not in hf:
                continue
            grp = hf[cat]
            for uid in sorted(grp.keys()):
                if uid == "protein_ids":
                    continue
                prot = grp[uid]
                if not isinstance(prot, h5py.Group):
                    continue
                N_res = int(prot.attrs.get("N_residues", -1))
                bin_label = get_n_res_bin(N_res)
                if bin_label is None:
                    continue

                L_norm = float(prot.attrs.get(norm_attr, np.nan))
                if not np.isfinite(L_norm) or L_norm <= 0:
                    continue

                try:
                    dg = prot[f"s{s}"][f"K{K}"][direction]
                except KeyError:
                    continue

                r   = dg["bin_r"][:] / L_norm      # ← 归一化
                rho = dg["bin_rho"][:]
                cnt = dg["bin_count"][:] if "bin_count" in dg else np.ones(len(r))

                valid = np.isfinite(r) & np.isfinite(rho) & (cnt >= 2)
                r_v, rho_v = r[valid], rho[valid]

                if len(r_v) == 0:
                    continue

                counts[bin_label] += 1
                for ri, rhoi in zip(r_v, rho_v):
                    r_key = round(ri / NORM_BIN_WIDTH) * NORM_BIN_WIDTH
                    if r_key > 0:
                        curve[bin_label][r_key].append(rhoi)

    return curve, counts


# ═══════════════════════════════════════════════════════════════════════════
#  绘图
# ═══════════════════════════════════════════════════════════════════════════
def plot_direction(curve, counts, direction, s, K, out_dir, xmax=None):
    """一个方向的 ρ(r) 分组均值曲线 + ±1 SD 色带（归一化距离版）。"""
    total_proteins = sum(counts.values())
    if total_proteins == 0:
        print(f"   ⚠ No data for '{direction}', skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(50 * MM, 36 * MM))

    bin_labels = [label for label, _, _ in N_RES_BINS]
    all_r = set()
    for bl in bin_labels:
        if counts[bl] == 0:
            continue
        bins = sorted(curve[bl].keys())
        r_mean, rho_mean, rho_sd = [], [], []
        for r_key in bins:
            vals = np.array(curve[bl][r_key])
            n = len(vals)
            if n < 3:
                continue
            r_mean.append(r_key)
            rho_mean.append(vals.mean())
            rho_sd.append(vals.std(ddof=1))

        r_mean = np.array(r_mean)
        rho_mean = np.array(rho_mean)
        rho_sd = np.array(rho_sd)

        lo_sd = rho_mean - rho_sd
        hi_sd = rho_mean + rho_sd

        # ± 1 SD 色带
        ax.fill_between(r_mean, lo_sd, hi_sd,
                        color=BIN_COLORS[bl], alpha=0.13, lw=0, zorder=2)
        # 均值线 + marker
        ax.plot(r_mean, rho_mean, "-", marker=BIN_MARKERS[bl],
                lw=1.4, color=BIN_COLORS[bl],
                markersize=4.0,
                markeredgewidth=0.3, markeredgecolor="black",
                zorder=4, label=bl)

        all_r.update(r_mean.tolist())

    # ── 参考线 ──
    ax.axhline(0, color=C_NULL, ls=":", lw=0.5, zorder=0)

    # ── 轴范围 ──
    r_vals = sorted(all_r)
    rmax = xmax if xmax is not None else (r_vals[-1] if r_vals else 3.0)
    rmax = np.ceil(rmax / 1.0) * 1.0
    ax.set_xlim(0, rmax)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    norm_label = direction.split("_norm_")[1]  # "xy" or "z"
    ax.set_xlabel(f"$\\rho_{{{norm_label}}}$")

    # y 轴范围仅基于 x 轴可见区间内的数据
    vis_lo, vis_hi = [], []
    for bl in bin_labels:
        if counts[bl] == 0:
            continue
        for r_key in sorted(curve[bl].keys()):
            if r_key > rmax:
                break
            vals = np.array(curve[bl][r_key])
            if len(vals) < 3:
                continue
            m, sd = vals.mean(), vals.std(ddof=1)
            vis_lo.append(m - sd)
            vis_hi.append(m + sd)

    y_lo = min(np.nanmin(vis_lo), -0.05) if vis_lo else -0.3
    y_hi = max(np.nanmax(vis_hi), 0.95) if vis_hi else 1.05
    pad = (y_hi - y_lo) * 0.06
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax.set_ylabel(DIR_LABELS.get(direction, direction), rotation=0, ha="right", va="center")

    ax.legend(loc="upper right", fontsize=6, frameon=True,
              handlelength=1.8, markerscale=0.8,
              borderpad=0.3, labelspacing=0.3)

    fname = f"corr_{direction}_s{s}.png"
    fig.savefig(out_dir / fname, dpi=600)
    plt.close(fig)
    return fname


# ═══════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=str, default=H5_PATH)
    ap.add_argument("--out", type=str, default=OUT_DIR)
    ap.add_argument("--s", type=int, default=S_DEFAULT)
    ap.add_argument("--K", type=int, default=K_DEFAULT)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 归一化距离版：xy 按 L_xy，z 按 L_z ──
    for direction, norm_key, norm_attr, xmax in [
            ("xy", "xy_norm_xy", "L_xy", 8.0),
            ("z",  "z_norm_z",   "L_z",  5.0)]:
        print(f"== {direction} (normalised by {norm_attr}) ==")
        curve_norm, counts_norm = load_corr_points_normalized(
            args.h5, args.s, args.K, direction, norm_attr=norm_attr)

        for bl in [label for label, _, _ in N_RES_BINS]:
            print(f"   {bl:12s}: {counts_norm[bl]:4d} proteins, "
                  f"{sum(len(v) for v in curve_norm[bl].values()):7d} points")
        print(f"   {'TOTAL':>12s}: {sum(counts_norm.values()):4d} proteins")

        fname = plot_direction(
            curve_norm, counts_norm, norm_key, args.s, args.K, out_dir,
            xmax=xmax)
        if fname:
            print(f"   -> {fname}")
        else:
            print(f"   ⚠ skipped (no data)")

    print(f"\n   Figures -> {out_dir}/")


if __name__ == "__main__":
    main()