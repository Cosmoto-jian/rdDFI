#!/usr/bin/env python3
"""
plot_eta_vs_s.py — 绘制 η = ξ_zero / L 随 s 变化的分类代表曲线。

读取 corr_length_final.py 输出的 xi_master_table.csv，
共出两张图：
  - η_xy = ξ_xy / L_xy（膜平面相关长度 / 膜平面半长轴归一化）
  - η_z  = ξ_z / L_z（膜法向相关长度 / 膜法向半长轴归一化）

每张图：
  - 横轴：s ∈ {1, 2, 4, 8, 16, 32, 64}
  - 4 条曲线（GPCR / large / medium / single_pass）
  - 每个点 = 该类别下所有蛋白 η 的均值
  - 误差棒 = ± 1 SE（标准误差，ddof=1）

Usage:
  python s_scan.py [--csv PATH] [--out DIR] [--K 200]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 参数 ──────────────────────────────────────────────────────────────────
CSV_PATH   = "/Volumes/x23/临时/rdDFI/data/process/xi_master_table_100_800.csv"
OUT_DIR    = "/Volumes/x23/临时/rdDFI/results/s_scan375_475"
S_VALUES   = [1, 2, 4, 8, 16, 32, 64]
K_DEFAULT  = 1000
CATEGORIES = ["GPCR", "large", "medium", "single_pass"]

CAT_LABELS  = {"GPCR": "7 TM(GPCR)", "large": "Large",
               "medium": "Medium", "single_pass": "Single-pass"}
CAT_COLORS  = {"GPCR": "#D62728", "large": "#1F77B4",
               "medium": "#2CA02C", "single_pass": "#9467BD"}
CAT_MARKERS = {"GPCR": "v", "large": "s", "medium": "^", "single_pass": "*"}

MM = 1.0 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 8,
    "legend.fontsize": 6, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": True, "legend.handlelength": 1.4,
    "axes.spines.top": True, "axes.spines.right": True,
    "lines.linewidth": 1.0, "lines.markersize": 4,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# 方向标签 — L_xy 归一化 → φ_xy
DIR_LABELS_XY = {
    "total": r"$\phi_{\mathrm{total}}$",
    "xy":    r"$\rho_{xy}^*$",
    "z":     r"$\rho_{z}^*$",
}
DIR_TITLES_XY = {
    "total": r"Total correlation ($\phi_{\mathrm{total}}$)",
    "xy":    r"In-plane correlation ($\rho_{xy}^*$)",
    "z":     r"Normal correlation ($\rho_{z}^*$)",
}
# 方向标签 — L_z 归一化 → φ_z
DIR_LABELS_Z = {
    "total": r"$\phi_{\mathrm{total}}$",
    "xy":    r"$\rho_{xy}^*$",
    "z":     r"$\rho_{z}^*$",
}
DIR_TITLES_Z = {
    "total": r"Total correlation ($\phi_{\mathrm{total}}$)",
    "xy":    r"In-plane correlation ($\rho_{xy}^*$)",
    "z":     r"Normal correlation ($\rho_{z}^*$)",
}


# ═══════════════════════════════════════════════════════════════════════════
#  计算：per-category 均值 ± SE
# ═══════════════════════════════════════════════════════════════════════════
def compute_eta_stats(df, direction, K, norm_attr="L_xy"):
    """对指定方向和 K，计算每个 (category, s) 的 η = ξ / norm_attr 均值 ± SE。

    返回 DataFrame: category, s, n, mean, se
    """
    sub = df[(df.direction == direction) & (df.K == K)].copy()
    sub = sub.dropna(subset=["xi_zero", norm_attr])
    sub = sub[sub[norm_attr] > 0]
    sub["eta"] = sub.xi_zero / sub[norm_attr]

    rows = []
    for cat in CATEGORIES:
        for s in S_VALUES:
            v = sub[(sub.category == cat) & (sub.s == s)]["eta"].dropna()
            n = len(v)
            rows.append({
                "category": cat, "s": s, "n": n,
                "mean": v.mean() if n > 0 else np.nan,
                "se": v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
#  绘图：均值 ± SE 误差棒
# ═══════════════════════════════════════════════════════════════════════════
def plot_eta_vs_s(eta_df, direction, out_dir, K, labels_dict=None, titles_dict=None,
                  norm_tag="Lxy", show_title=True, show_legend=True):
    """对一个方向画 η vs s：均值 ± SE 误差棒。"""
    fig, ax = plt.subplots(figsize=(60 * MM, 50 * MM))

    for cat in CATEGORIES:
        d = eta_df[eta_df.category == cat].sort_values("s")
        ok = d["mean"].notna()
        d = d[ok]
        if len(d) == 0:
            continue

        n_rep = d.n.iloc[0]
        yerr = d["se"].values

        ax.errorbar(d.s, d["mean"], yerr=yerr,
                    fmt="-", marker=CAT_MARKERS[cat],
                    color=CAT_COLORS[cat], markersize=4.5,
                    markeredgewidth=0.4, markeredgecolor="black",
                    lw=1.2, capsize=2.5, capthick=0.5,
                    elinewidth=0.6, zorder=3,
                    label=CAT_LABELS.get(cat, cat))

    ax.set_xscale("log", base=2)
    ax.set_xticks(S_VALUES)
    ax.set_xticklabels([str(v) for v in S_VALUES])
    ax.set_xlabel(r"$s$")
    if labels_dict:
        ax.set_ylabel(labels_dict[direction], rotation=0, ha="right", va="center")

    # 标题含 K 信息
    K_label = "full mode" if K >= 1000 else f"K = {K}"
    if show_title and titles_dict:
        ax.set_title(titles_dict[direction] + f"    ({K_label})", pad=4, fontsize=10)
    if show_legend:
        ax.legend(loc="best", fontsize=6, frameon=True, handlelength=2)

    fname = f"eta_{direction}_norm_{norm_tag}_K{K}_vs_s.png"
    fig.savefig(out_dir / fname, dpi=600)
    plt.close(fig)
    return fname


# ═══════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=CSV_PATH,
                    help="xi_master_table.csv 路径")
    ap.add_argument("--out", type=str, default=OUT_DIR,
                    help="输出目录")
    ap.add_argument("--K", type=int, default=K_DEFAULT,
                    help="模态截断数")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== Loading ==")
    df = pd.read_csv(args.csv)
    print(f"   {len(df)} rows (all), "
          f"{df.pid.nunique()} proteins (all), "
          f"K = {args.K}")

    # ── N_res filtering ──
    if "N_res" in df.columns:
        df = df[(df["N_res"] >= 375) & (df["N_res"] <= 475)]
        print(f"   after N_res ∈ [375,475]: {len(df)} rows, "
              f"{df.pid.nunique()} proteins")

    all_stats = []

    # ── L_xy 归一化 ──
    for direction in ["xy"]:
        print(f"\n== {direction} / L_xy ==")
        eta_df = compute_eta_stats(df, direction, args.K, norm_attr="L_xy")
        all_stats.append(eta_df.assign(direction=direction, norm="L_xy"))

        for cat in CATEGORIES:
            d = eta_df[eta_df.category == cat]
            ns = d.n.values
            n0 = ns[0] if len(ns) > 0 else 0
            m_s1 = d[d.s == 1]["mean"].values
            m_s64 = d[d.s == 64]["mean"].values
            s1_str = f"{m_s1[0]:.3f}" if len(m_s1) > 0 else "N/A"
            s64_str = f"{m_s64[0]:.3f}" if len(m_s64) > 0 else "N/A"
            print(f"   {CAT_LABELS.get(cat, cat):12s}: n = {n0:4d},  "
                  f"η(s=1) = {s1_str},  "
                  f"η(s=64) = {s64_str}")

        fname = plot_eta_vs_s(eta_df, direction, out_dir, args.K,
                              DIR_LABELS_XY, DIR_TITLES_XY, "Lxy")
        print(f"   -> {fname}")

    # ── L_z 归一化 ──
    for direction in ["z"]:
        print(f"\n== {direction} / L_z ==")
        eta_df = compute_eta_stats(df, direction, args.K, norm_attr="L_z")
        all_stats.append(eta_df.assign(direction=direction, norm="L_z"))

        for cat in CATEGORIES:
            d = eta_df[eta_df.category == cat]
            ns = d.n.values
            n0 = ns[0] if len(ns) > 0 else 0
            m_s1 = d[d.s == 1]["mean"].values
            m_s64 = d[d.s == 64]["mean"].values
            s1_str = f"{m_s1[0]:.3f}" if len(m_s1) > 0 else "N/A"
            s64_str = f"{m_s64[0]:.3f}" if len(m_s64) > 0 else "N/A"
            print(f"   {CAT_LABELS.get(cat, cat):12s}: n = {n0:4d},  "
                  f"η(s=1) = {s1_str},  "
                  f"η(s=64) = {s64_str}")

        fname = plot_eta_vs_s(eta_df, direction, out_dir, args.K,
                              DIR_LABELS_Z, DIR_TITLES_Z, "Lz")
        print(f"   -> {fname}")

    # 汇总 CSV
    all_df = pd.concat(all_stats, ignore_index=True)
    csv_out = out_dir / "eta_vs_s_stats.csv"
    all_df.to_csv(csv_out, index=False)
    print(f"\n   Stats -> {csv_out}")
    print(f"   Figures -> {out_dir}/")


if __name__ == "__main__":
    main()
