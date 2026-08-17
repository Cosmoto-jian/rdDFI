#!/usr/bin/env python3
"""
xyz_length_scan.py — GPCR 蛋白 ρ(r) 对比图：不同 K 值（50/100/200/1000）。

固定 s=16，只取 GPCR 类别中长度 375–475 的蛋白，
对 xy 和 z 方向各出一张图，对比 4 条 K 值均值曲线。

直接从 HDF5 读取 bin_r / bin_rho 原始数据。
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
OUT_DIR    = "results/cor_L_results"
S_DEFAULT  = 16
N_RES_MIN  = 375
N_RES_MAX  = 475

# K 值列表：20, 50, 100, 200, 1000
K_VALUES = [20, 50, 100, 200, 1000]

K_COLORS   = {20: "#D4A5D0", 50: "#457B9D", 100: "#E63946", 200: "#2A9D8F", 1000: "#F4A261"}
K_MARKERS  = {20: "v", 50: "o", 100: "s", 200: "^", 1000: "D"}
K_LABELS   = {20: "K = 20", 50: "K = 50", 100: "K = 100", 200: "K = 200", 1000: "K = 1000"}

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

DIR_LABELS = {
    "xy": r"$\phi_{xy}$",
    "z":  r"$\phi_{z}$",
}
DIR_TITLES = {
    "xy": "In-plane (xy)",
    "z":  "Normal (z)",
}
DIR_COLORS = {"xy": "#457B9D", "z": "#E63946"}
DIR_MARKERS = {"xy": "^", "z": "s"}


# ═══════════════════════════════════════════════════════════════════════════
#  数据提取：只读 GPCR，按 K 值分组
# ═══════════════════════════════════════════════════════════════════════════
def load_corr_points(h5_path, s, K_values, direction):
    """读取 GPCR 蛋白的 (r, rho) 数据，按 K 值分组。

    只取 N_residues ∈ [375, 475] 的蛋白。

    返回:
      curve  : {K_label: {r_key: [rho_values]}}  用于均值曲线
      counts : {K_label: n_proteins}
    """
    curve  = {k: defaultdict(list) for k in K_values}
    counts = {k: 0 for k in K_values}

    with h5py.File(h5_path, "r") as hf:
        if "GPCR" not in hf:
            print("   ⚠ 'GPCR' group not found in HDF5")
            return curve, counts

        grp = hf["GPCR"]
        for uid in sorted(grp.keys()):
            if uid == "protein_ids":
                continue
            prot = grp[uid]
            if not isinstance(prot, h5py.Group):
                continue
            N_res = int(prot.attrs.get("N_residues", -1))
            if N_res < N_RES_MIN or N_res > N_RES_MAX:
                continue

            for K in K_values:
                try:
                    dg = prot[f"s{s}"][f"K{K}"][direction]
                except KeyError:
                    continue

                r   = dg["bin_r"][:]
                rho = dg["bin_rho"][:]
                cnt = dg["bin_count"][:] if "bin_count" in dg else np.ones(len(r))

                # 只取 count >= 2 的有效 bin
                valid = np.isfinite(r) & np.isfinite(rho) & (cnt >= 2)
                r_v, rho_v = r[valid], rho[valid]

                if len(r_v) == 0:
                    continue

                counts[K] += 1
                for ri, rhoi in zip(r_v, rho_v):
                    r_key = round(ri, 1)
                    curve[K][r_key].append(rhoi)

    return curve, counts


# ═══════════════════════════════════════════════════════════════════════════
#  绘图
# ═══════════════════════════════════════════════════════════════════════════
def plot_direction(curve, counts, direction, s, out_dir, ylim=None, show_title=True):
    """一个方向的 ρ(r) K 值对比均值曲线（4 条曲线）。"""
    total_proteins = max(counts.values())
    if total_proteins == 0:
        print(f"   ⚠ No data for '{direction}', skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(50 * MM, 36 * MM))

    all_r = set()
    for K in K_VALUES:
        if counts[K] == 0:
            continue
        bins = sorted(curve[K].keys())
        r_mean, rho_mean = [], []
        for r_key in bins:
            vals = np.array(curve[K][r_key])
            n = len(vals)
            if n < 3:
                continue
            r_mean.append(r_key)
            rho_mean.append(vals.mean())

        r_mean = np.array(r_mean)
        rho_mean = np.array(rho_mean)

        # 均值线 + marker（无 SD 色带）
        ax.plot(r_mean, rho_mean, "-",
                marker=K_MARKERS[K],
                lw=1.4,
                color=K_COLORS[K],
                markersize=4.0,
                markeredgewidth=0.3,
                markeredgecolor="black",
                zorder=4,
                label=f"{K_LABELS[K]}")

        all_r.update(r_mean.tolist())

    # ── 参考线 ──
    ax.axhline(0, color=C_NULL, ls=":", lw=0.5, zorder=0)

    # ── x 轴范围 ──
    r_vals = sorted(all_r)
    rmax = r_vals[-1] if r_vals else 85.0
    rmax = min(np.ceil(rmax / 10) * 10, 90)
    ax.set_xlim(0, rmax)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel("$r_{ij}$ (Å)")

    # ── y 轴范围（仅基于 x 轴可见区间） ──
    vis_vals = []
    for K in K_VALUES:
        if counts[K] == 0:
            continue
        for r_key in sorted(curve[K].keys()):
            if r_key > rmax:
                break
            vals = np.array(curve[K][r_key])
            if len(vals) < 3:
                continue
            vis_vals.extend(vals.tolist())

    if vis_vals:
        y_lo = np.nanmin(vis_vals)
        y_hi = max(np.nanmax(vis_vals), 0.95)
    else:
        y_lo, y_hi = -0.3, 1.05
    pad = (y_hi - y_lo) * 0.06
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_ylabel(DIR_LABELS[direction], rotation=0, ha="right", va="center")
    if show_title:
        ax.set_title(f"{DIR_TITLES.get(direction, direction)}    "
                     f"(s = {s},  GPCR)",
                     pad=4, fontsize=10)

    ax.legend(loc="upper right", fontsize=6, frameon=True,
              handlelength=1.8, markerscale=0.8,
              borderpad=0.3, labelspacing=0.3)

    fname = f"corr_{direction}_s{s}.png"
    fig.savefig(out_dir / fname, dpi=600)
    plt.close(fig)
    return fname


# ═══════════════════════════════════════════════════════════════════════════
#  绘图：xy vs z 对比（K=1000）
# ═══════════════════════════════════════════════════════════════════════════
def plot_xy_z_compare(curve_xy, curve_z, counts, s, out_dir, cat_label="GPCR", ylim=None, show_title=True):
    """K=1000 下 ρ_xy 与 ρ_z 对比（均值曲线）。"""
    total = max(counts.values())
    if total == 0:
        print("   ⚠ No data for xy/z comparison, skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(50 * MM, 36 * MM))

    for direction, curve in [("xy", curve_xy), ("z", curve_z)]:
        bins = sorted(curve.keys())
        r_mean, rho_mean = [], []
        for r_key in bins:
            vals = np.array(curve[r_key])
            n = len(vals)
            if n < 3:
                continue
            r_mean.append(r_key)
            rho_mean.append(vals.mean())

        r_mean = np.array(r_mean)
        rho_mean = np.array(rho_mean)

        ax.plot(r_mean, rho_mean, "-",
                marker=DIR_MARKERS[direction],
                lw=1.4, color=DIR_COLORS[direction],
                markersize=4.0,
                markeredgewidth=0.3, markeredgecolor="black",
                zorder=4,
                label=DIR_LABELS[direction])

    # ── 参考线 ──
    ax.axhline(0, color=C_NULL, ls=":", lw=0.5, zorder=0)

    # ── x 轴范围 ──
    all_r = set()
    for curve in [curve_xy, curve_z]:
        all_r.update(curve.keys())
    r_vals = sorted(all_r)
    rmax = r_vals[-1] if r_vals else 85.0
    rmax = min(np.ceil(rmax / 10) * 10, 90)
    ax.set_xlim(0, rmax)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel("$r_{ij}$ (Å)")

    # ── y 轴范围 ──
    vis_vals = []
    for curve in [curve_xy, curve_z]:
        for r_key in curve:
            if r_key > rmax:
                break
            vals = np.array(curve[r_key])
            if len(vals) < 3:
                continue
            vis_vals.extend(vals.tolist())
    if vis_vals:
        y_lo = np.nanmin(vis_vals)
        y_hi = max(np.nanmax(vis_vals), 0.95)
    else:
        y_lo, y_hi = -0.3, 1.05
    pad = (y_hi - y_lo) * 0.06
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_ylabel(r"$\phi$", rotation=0, ha="right", va="center")
    if show_title:
        ax.set_title(f"xy vs z    (s = {s},  K = 1000,  {cat_label},  n = {total})",
                     pad=4, fontsize=10)

    ax.legend(loc="upper right", fontsize=6, frameon=True,
              handlelength=1.8, markerscale=0.8,
              borderpad=0.3, labelspacing=0.3)

    suffix = "" if cat_label == "GPCR" else "_allcats"
    fname = f"corr_xy_vs_z_s{s}{suffix}.png"
    fig.savefig(out_dir / fname, dpi=600)
    plt.close(fig)
    return fname


# ═══════════════════════════════════════════════════════════════════════════
#  数据提取：全部四类蛋白汇总（不区分类别），单 K 值
# ═══════════════════════════════════════════════════════════════════════════
def load_corr_all_cats(h5_path, s, K, direction):
    """读取全部四类蛋白的 (r, rho) 数据，汇总为一个池。

    只取 N_residues ∈ [375, 475] 的蛋白。

    返回:
      curve  : {r_key: [rho_values]}
      count  : n_proteins (total)
    """
    CATEGORIES_ALL = ["GPCR", "large", "medium", "single_pass"]
    curve = defaultdict(list)
    count = 0

    with h5py.File(h5_path, "r") as hf:
        for cat in CATEGORIES_ALL:
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
                if N_res < N_RES_MIN or N_res > N_RES_MAX:
                    continue

                try:
                    dg = prot[f"s{s}"][f"K{K}"][direction]
                except KeyError:
                    continue

                r   = dg["bin_r"][:]
                rho = dg["bin_rho"][:]
                cnt = dg["bin_count"][:] if "bin_count" in dg else np.ones(len(r))

                valid = np.isfinite(r) & np.isfinite(rho) & (cnt >= 2)
                r_v, rho_v = r[valid], rho[valid]

                if len(r_v) == 0:
                    continue

                count += 1
                for ri, rhoi in zip(r_v, rho_v):
                    r_key = round(ri, 1)
                    curve[r_key].append(rhoi)

    return curve, count


# ═══════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=str, default=H5_PATH)
    ap.add_argument("--out", type=str, default=OUT_DIR)
    ap.add_argument("--s", type=int, default=S_DEFAULT)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for direction in ["xy", "z"]:
        print(f"== {direction} (GPCR, s={args.s}) ==")
        curve, counts = load_corr_points(
            args.h5, args.s, K_VALUES, direction)

        for K in K_VALUES:
            n_pts = sum(len(v) for v in curve[K].values())
            print(f"   {K_LABELS[K]:12s}: {counts[K]:4d} proteins, "
                  f"{n_pts:7d} points")

        fname = plot_direction(curve, counts, direction, args.s, out_dir,
                               ylim=(-0.5, 1) if direction == "xy" else (-0.3, 1),
                               show_title=False)
        if fname:
            print(f"   -> {fname}")
        else:
            print(f"   ⚠ skipped (no data)")

    # ── xy vs z 对比图（s=16, K=1000） ──
    s_val = 16
    print(f"== xy vs z comparison (GPCR, s={s_val}, K=1000) ==")
    curve_xy, counts_xy = load_corr_points(
        args.h5, s_val, [1000], "xy")
    curve_z, counts_z = load_corr_points(
        args.h5, s_val, [1000], "z")

    print(f"   {DIR_LABELS['xy']:24s}: {counts_xy[1000]:4d} proteins, "
          f"{sum(len(v) for v in curve_xy[1000].values()):7d} points")
    print(f"   {DIR_LABELS['z']:24s}: {counts_z[1000]:4d} proteins, "
          f"{sum(len(v) for v in curve_z[1000].values()):7d} points")

    fname = plot_xy_z_compare(
        curve_xy[1000], curve_z[1000],
        {"xy": counts_xy[1000], "z": counts_z[1000]},
        s_val, out_dir,
        ylim=(-0.5, 1),
        show_title=True)
    if fname:
        print(f"   -> {fname}")
    else:
        print(f"   ⚠ skipped (no data)")

    # ── 全部四类汇总 xy vs z 对比图（s=16, K=1000） ──
    print(f"== xy vs z comparison (all 4 categories pooled, s=16, K=1000) ==")
    curve_xy_all, count_xy_all = load_corr_all_cats(
        args.h5, 16, 1000, "xy")
    curve_z_all, count_z_all = load_corr_all_cats(
        args.h5, 16, 1000, "z")

    n_pts_xy = sum(len(v) for v in curve_xy_all.values())
    n_pts_z  = sum(len(v) for v in curve_z_all.values())
    print(f"   {DIR_LABELS['xy']:24s}: {count_xy_all:4d} proteins, "
          f"{n_pts_xy:7d} points")
    print(f"   {DIR_LABELS['z']:24s}: {count_z_all:4d} proteins, "
          f"{n_pts_z:7d} points")

    fname = plot_xy_z_compare(
        curve_xy_all, curve_z_all,
        {"all": count_xy_all},
        16, out_dir, cat_label="all 4 categories", ylim=(-0.5, 1))
    if fname:
        print(f"   -> {fname}")
    else:
        print(f"   ⚠ skipped (no data)")

    print(f"\n   Figures -> {out_dir}/")


if __name__ == "__main__":
    main()
