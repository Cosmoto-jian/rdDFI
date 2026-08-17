#!/usr/bin/env python3
"""
cor_length_scan.py — case_assembly 蛋白 ρ(r) 对比图（单 K 值）。

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
H5_PATH    = "data/raw/case_assembly/case_assembly.h5"
OUT_DIR    = "results/case_assembly/cor_L_results"
S_DEFAULT  = 16
K_FIX      = 1000   # 单 K 值（原 K 扫描已删除）

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
DIR_COLORS = {"xy": "#457B9D", "z": "#E63946"}
DIR_MARKERS = {"xy": "^", "z": "s"}


# ═══════════════════════════════════════════════════════════════════════════
#  数据提取：单 K 值，按类别组读取
# ═══════════════════════════════════════════════════════════════════════════
def load_corr_group(h5_path, s, K, direction, group_name):
    """读取 H5 中指定类别组（single_chain / assembly）的 (r, rho) 数据。

    返回:
      curve  : {r_key: [rho_values]}
      count  : n_proteins
    """
    curve = defaultdict(list)
    count = 0

    with h5py.File(h5_path, "r") as hf:
        if group_name not in hf:
            print(f"   ⚠ '{group_name}' group not found in HDF5")
            return curve, count

        grp = hf[group_name]
        for uid in sorted(grp.keys()):
            if uid == "protein_ids":
                continue
            prot = grp[uid]
            if not isinstance(prot, h5py.Group):
                continue

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

            count += 1
            for ri, rhoi in zip(r_v, rho_v):
                r_key = round(ri, 1)
                curve[r_key].append(rhoi)

    return curve, count


def load_corr_points(h5_path, s, K, direction):
    """读取 single_chain 类别的 (r, rho) 数据（原 GPCR 类别）。"""
    return load_corr_group(h5_path, s, K, direction, "single_chain")


def load_corr_all_cats(h5_path, s, K, direction):
    """读取 assembly 类别的 (r, rho) 数据（原"全部四类汇总"）。"""
    return load_corr_group(h5_path, s, K, direction, "assembly")


# ═══════════════════════════════════════════════════════════════════════════
#  过零点：曲线第一次穿过 0 时的 r_ij
# ═══════════════════════════════════════════════════════════════════════════
def mean_curve(curve):
    """{r_key: [rho]} → (r_mean, rho_mean)，按 r 升序。"""
    bins = sorted(curve.keys())
    r_mean, rho_mean = [], []
    for r_key in bins:
        vals = np.array(curve[r_key])
        if len(vals) < 1:
            continue
        r_mean.append(r_key)
        rho_mean.append(vals.mean())
    return np.array(r_mean, dtype=float), np.array(rho_mean, dtype=float)


def first_zero_crossing(curve):
    """曲线第一次过零点的 r_ij（相邻 bin 中心间线性插值）；无过零点返回 None。"""
    r_mean, rho_mean = mean_curve(curve)
    for i in range(len(rho_mean)):
        if rho_mean[i] == 0.0:
            return float(r_mean[i])
        if i + 1 < len(rho_mean) and rho_mean[i] * rho_mean[i + 1] < 0:
            t = -rho_mean[i] / (rho_mean[i + 1] - rho_mean[i])
            return float(r_mean[i] + t * (r_mean[i + 1] - r_mean[i]))
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  nice 刻度：轴端吸附到 1/2/5×10^n 倍数
# ═══════════════════════════════════════════════════════════════════════════
def _nice_step(span, target_ticks=6):
    """返回 1/2/5×10^n 步长，使轴上有约 target_ticks 个大刻度。"""
    raw = span / target_ticks
    pow10 = 10.0 ** np.floor(np.log10(raw))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * pow10 >= raw:
            return m * pow10
    return 10.0 * pow10


def _snap_lo(v, step):
    """向下吸附到 step 的倍数（含容差，避免浮点误差多减一格）。"""
    return np.floor(v / step + 1e-9) * step


def _snap_hi(v, step):
    """向上吸附到 step 的倍数。"""
    return np.ceil(v / step - 1e-9) * step


# ═══════════════════════════════════════════════════════════════════════════
#  绘图：xy vs z 对比（单 K 值）
# ═══════════════════════════════════════════════════════════════════════════
def plot_xy_z_compare(curve_xy, curve_z, counts, s, K, out_dir,
                      cat_label="single_chain", ylim=None, show_title=True):
    """固定 K 下 ρ_xy 与 ρ_z 对比（均值曲线）。"""
    total = max(counts.values())
    if total == 0:
        print("   ⚠ No data for xy/z comparison, skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(50 * MM, 36 * MM))

    for direction, curve in [("xy", curve_xy), ("z", curve_z)]:
        # 每个类别只有一个蛋白：每个 r_key 仅 1 个值，均值即其本身
        r_mean, rho_mean = mean_curve(curve)

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
    # 吸附到 10 的倍数（+2.5 Å 半 bin 宽），完整显示最远分箱及端点 marker
    rmax = _snap_hi(rmax + 2.5, 10)
    ax.set_xlim(0, rmax)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel("$r_{ij}$ (Å)")

    # ── y 轴范围（按数据自动截断，轴端吸附到 nice 刻度倍数）──
    vis_vals = []
    for curve in [curve_xy, curve_z]:
        for r_key in sorted(curve):   # key 无排序保证，遍历全部后再筛选
            if r_key > rmax:
                continue
            vals = np.array(curve[r_key])
            if len(vals) < 1:
                continue
            vis_vals.extend(vals.tolist())
    if vis_vals:
        y_lo = min(np.nanmin(vis_vals), 0.0)   # 包含 0，零参考线保持在轴内
        y_hi = max(np.nanmax(vis_vals), 0.0)
        pad = max((y_hi - y_lo) * 0.05, 0.03)  # 端点 marker（4 pt）余量
        step = _nice_step(y_hi - y_lo + 2 * pad)
        y_lo = _snap_lo(y_lo - pad, step)
        y_hi = _snap_hi(y_hi + pad, step)
    else:
        y_lo, y_hi, step = -0.3, 1.05, 0.5
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    minor = step / 2 if step <= 0.2 else step / 5
    ax.yaxis.set_minor_locator(MultipleLocator(minor))

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_ylabel(r"$\phi$", rotation=0, ha="right", va="center")
    if show_title:
        ax.set_title(f"xy vs z    (s = {s},  K = {K},  {cat_label},  n = {total})",
                     pad=4, fontsize=4)

    ax.legend(loc="upper right", fontsize=6, frameon=True,
              handlelength=1.8, markerscale=0.8,
              borderpad=0.3, labelspacing=0.3)

    suffix = "" if cat_label == "single_chain" else f"_{cat_label}"
    fname = f"corr_xy_vs_z_s{s}{suffix}.png"
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
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── single_chain xy vs z 对比图 ──
    print(f"== xy vs z comparison (single_chain, s={args.s}, K={K_FIX}) ==")
    curve_xy, count_xy = load_corr_points(args.h5, args.s, K_FIX, "xy")
    curve_z, count_z = load_corr_points(args.h5, args.s, K_FIX, "z")

    print(f"   {DIR_LABELS['xy']:24s}: {count_xy:4d} proteins, "
          f"{sum(len(v) for v in curve_xy.values()):7d} points")
    print(f"   {DIR_LABELS['z']:24s}: {count_z:4d} proteins, "
          f"{sum(len(v) for v in curve_z.values()):7d} points")

    fname = plot_xy_z_compare(
        curve_xy, curve_z,
        {"xy": count_xy, "z": count_z},
        args.s, K_FIX, out_dir, cat_label="single_chain",
        show_title=True)
    if fname:
        print(f"   -> {fname}")
    else:
        print(f"   ⚠ skipped (no data)")

    # ── assemblyxy vs z 对比图 ──
    print(f"== xy vs z comparison (assembly, s={args.s}, K={K_FIX}) ==")
    curve_xy_all, count_xy_all = load_corr_all_cats(args.h5, args.s, K_FIX, "xy")
    curve_z_all, count_z_all = load_corr_all_cats(args.h5, args.s, K_FIX, "z")

    n_pts_xy = sum(len(v) for v in curve_xy_all.values())
    n_pts_z  = sum(len(v) for v in curve_z_all.values())
    print(f"   {DIR_LABELS['xy']:24s}: {count_xy_all:4d} proteins, "
          f"{n_pts_xy:7d} points")
    print(f"   {DIR_LABELS['z']:24s}: {count_z_all:4d} proteins, "
          f"{n_pts_z:7d} points")

    fname = plot_xy_z_compare(
        curve_xy_all, curve_z_all,
        {"all": count_xy_all},
        args.s, K_FIX, out_dir, cat_label="assembly")
    if fname:
        print(f"   -> {fname}")
    else:
        print(f"   ⚠ skipped (no data)")

    # ── 第一次过零点的 r_ij 汇总 → txt ──
    zero_rows = [
        ("single_chain", first_zero_crossing(curve_xy),
                          first_zero_crossing(curve_z)),
        ("assembly",     first_zero_crossing(curve_xy_all),
                          first_zero_crossing(curve_z_all)),
    ]
    zero_file = out_dir / "first_zero_crossing.txt"
    with open(zero_file, "w") as f:
        f.write(f"First zero-crossing r_ij (Å)   (s = {args.s},  K = {K_FIX})\n\n")
        f.write(f"{'category':<14s} {'phi_xy':>10s} {'phi_z':>10s}\n")
        for cat, r_xy, r_z in zero_rows:
            s_xy = f"{r_xy:.1f}" if r_xy is not None else "n/a"
            s_z  = f"{r_z:.1f}"  if r_z  is not None else "n/a"
            f.write(f"{cat:<14s} {s_xy:>10s} {s_z:>10s}\n")
    print(f"   -> {zero_file.name}")

    print(f"\n   Figures -> {out_dir}/")


if __name__ == "__main__":
    main()
