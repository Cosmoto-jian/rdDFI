#!/usr/bin/env python3
"""
Generate Figure 4 (exact copy of heat.py make_figure_1) and Figure 5 (exact copy of first.py make_figure_2).
Venn diagram numbers have no white background (bbox removed).
- Updated: Changed dpDFI to rdDFI, reduced font size, and placed labels inside the top of each circle.
- Updated: Input paths and OUT_DIR set to current script directory (./).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import skew
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines
import matplotlib.patches as patches
import math

# =====================================================================
# 0. Configuration & Paths (All set to current script dir ./)
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DFI_DIR = os.path.join(SCRIPT_DIR, "results", "DFI")
MUT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "mutation")
OUT_DIR = SCRIPT_DIR

Z_THRESH = -1.0

GPCR_TM_REGIONS = {
    "aa2ar": [(8, 32), (43, 66), (79, 108), (119, 142), (173, 202), (235, 258), (267, 290)],
    "adrb2": [(29, 60), (67, 96), (103, 136), (147, 171), (197, 229), (267, 298), (305, 331)],
    "cxcr4": [(39, 68), (78, 106), (112, 143), (157, 180), (203, 229), (240, 268), (280, 305)],
    "aa1r":  [(12, 32), (48, 68), (78, 98), (128, 147), (174, 196), (237, 259), (268, 289)],
    "aa2br": [(9, 33), (44, 67), (79, 101), (122, 144), (179, 203), (236, 259), (268, 291)],
    "acm1":  [(23, 48), (63, 84), (105, 126), (143, 164), (187, 210), (351, 372), (385, 407)],
    "ccr1":  [(35, 60), (73, 95), (108, 129), (151, 175), (198, 223), (240, 264), (282, 305)],
    "ccr5":  [(31, 58), (69, 89), (103, 124), (142, 166), (199, 218), (236, 261), (271, 295)],
    "cxcr1": [(39, 65), (76, 96), (111, 132), (153, 176), (198, 220), (243, 267), (277, 302)],
    "glp1r": [(145, 165), (174, 194), (228, 248), (271, 291), (317, 337), (350, 370), (383, 403)],
    "mc4r":  [(44, 64), (77, 97), (115, 135), (154, 174), (196, 216), (247, 267), (280, 300)],
}

# =====================================================================
# 1. Visual System
# =====================================================================
METRIC_STYLE = {
    "DFI":             {"color": "#3B82F6", "ls": "-", "label": "DFI"},
    "DFI_membrane":    {"color": "#1E3A8A", "ls": "-", "label": "imDFI"},
    "H":               {"color": "#FF0000", "ls": "-", "label": "rdDFI"},
}
RUG_METRICS = ["DFI", "DFI_membrane", "H"]

PALETTE = {
    "figure_bg": "#FFFFFF", "tm_bg": "#E2E8F0", "grid": "#E2E8F0",
    "text": "#000000", "mut_tm": "#94A3B8", "mut_nontm": "#CBD5E1",
}

RC_PARAMS = {
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "axes.linewidth": 1.0,
    "figure.facecolor": PALETTE["figure_bg"], "axes.facecolor": PALETTE["figure_bg"],
}
matplotlib.rcParams.update(RC_PARAMS)

CUSTOM_COLORS = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
CMAP = mcolors.LinearSegmentedColormap.from_list("custom_green", CUSTOM_COLORS, N=256)

# =====================================================================
# 2. Helper functions
# =====================================================================
def load_protein_data(protein_key):
    dfi_csv = os.path.join(DFI_DIR, f"{protein_key}.csv")
    mut_csv = os.path.join(MUT_RESULTS_DIR, protein_key, f"{protein_key}_mutation_summary.csv")
    if not os.path.exists(dfi_csv) or not os.path.exists(mut_csv):
        dfi_csv = f"{protein_key}.csv"
        mut_csv = f"{protein_key}_mutation_summary.csv"
        if not os.path.exists(dfi_csv) or not os.path.exists(mut_csv):
            return None, None, None

    dfi = pd.read_csv(dfi_csv).sort_values("ResI").reset_index(drop=True)
    valid = (dfi["DFI_XY"] > 0) & (dfi["DFI_Z"] > 0) & (dfi["DFI"] > 0) & (dfi["DFI_membrane"] > 0)
    dfi = dfi[valid].copy()
    dfi["H"] = np.log(dfi["DFI_Z"]) - np.log(dfi["DFI_XY"])

    mut_summary = pd.read_csv(mut_csv)
    mutated_positions = set(mut_summary["Position"].dropna().astype(int).unique())

    pos_mut_count = {}
    for _, row in mut_summary.iterrows():
        p = int(row["Position"])
        pos_mut_count[p] = pos_mut_count.get(p, 0) + 1

    return dfi, mutated_positions, pos_mut_count

def evaluate_selection(dfi_tm, metric, mutated_positions_tm, z_thresh=Z_THRESH):
    n_total = len(dfi_tm)
    s = dfi_tm[metric].astype(float)
    std = s.std(ddof=0)
    if std > 0:
        z = (s - s.mean()) / std
    else:
        z = s * 0.0
    selected_positions = dfi_tm[z < z_thresh]["ResI"].astype(int).tolist()
    n_select = len(selected_positions)
    hits = sorted(set(selected_positions) & mutated_positions_tm)
    misses = sorted(set(selected_positions) - mutated_positions_tm)
    precision = len(hits) / n_select if n_select else 0.0
    recall = len(hits) / len(mutated_positions_tm) if mutated_positions_tm else 0.0
    return {
        "n_total": n_total, "n_select": n_select,
        "n_mutated": len(mutated_positions_tm),
        "hits": hits, "misses": misses, "n_hits": len(hits),
        "precision": precision, "recall": recall,
    }

def _draw_tm_shading_and_labels(ax, tm_regions, x_min, x_max, show_labels=True):
    if not tm_regions: return
    for i, (a, b) in enumerate(tm_regions):
        a_c, b_c = max(a, x_min), min(b, x_max)
        if a_c < b_c:
            ax.axvspan(a_c, b_c, facecolor=PALETTE["tm_bg"], alpha=0.6, zorder=0, edgecolor="none")
            if show_labels:
                ax.text((a_c + b_c) / 2, 0.98, f"TM{i+1}", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=10, fontweight='bold', color="black", zorder=5)

def apply_black_border(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

# =====================================================================
# 3. Figure 4
# =====================================================================
def add_zoom_inset(ax, sweep_trim, ratios, ratio_colors, col, x_lim, arrow_color, arrow_xy, arrow_xytext):
    y_vals_all = []
    for ratio in ratios:
        full_sub = sweep_trim[sweep_trim["ratio"] == ratio]
        if len(full_sub) == 0: continue
        if col == "H":
            sm = full_sub["H"].rolling(window=5, center=True, min_periods=1).mean()
        else:
            sm = full_sub[col].rolling(window=5, center=True, min_periods=1).mean() * 1000.0
        mask = (full_sub["ResI"] >= x_lim[0]) & (full_sub["ResI"] <= x_lim[1])
        y_vals_all.extend(sm[mask].dropna().tolist())
    if y_vals_all:
        ymin, ymax = min(y_vals_all), max(y_vals_all)
        ypad = (ymax - ymin) * 0.15
        y_lim = (ymin - ypad, ymax + ypad)
    else:
        y_lim = (0, 1)

    axins = ax.inset_axes([1.03, 0.0, 0.22, 1.0])
    axins.set_facecolor(PALETTE["figure_bg"])
    for ratio in ratios:
        full_sub = sweep_trim[sweep_trim["ratio"] == ratio]
        if col == "H":
            sm = full_sub["H"].rolling(window=5, center=True, min_periods=1).mean()
        else:
            sm = full_sub[col].rolling(window=5, center=True, min_periods=1).mean() * 1000.0
        axins.plot(full_sub["ResI"], sm, color=ratio_colors[ratio], ls="-", lw=1.5, alpha=1.0, zorder=3)
    axins.set_xlim(x_lim)
    axins.set_ylim(y_lim)
    axins.set_xticks([])
    axins.set_yticks([])
    rect, connectors = ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.8, linewidth=1.0)
    for c in connectors:
        c.set_visible(False)
    connectors[2].set_visible(True)
    connectors[2].set_linestyle("--")
    connectors[3].set_visible(True)
    connectors[3].set_linestyle("--")
    axins.annotate('', xy=arrow_xy, xytext=arrow_xytext, xycoords='axes fraction',
                   arrowprops=dict(facecolor=arrow_color, edgecolor='none', width=6, headwidth=14), zorder=5)
    apply_black_border(axins)

def plot_figure4(protein_key="ccr1", sweep_csv_name="ccr1_ratio_sweep.csv"):
    sweep_csv = os.path.join(SCRIPT_DIR, sweep_csv_name)
    print("Generating Figure 4...")
    if not os.path.exists(sweep_csv):
        print(f"Error: {sweep_csv} not found!")
        return
    sweep_data = pd.read_csv(sweep_csv)
    ratios = [1, 2, 4, 8, 16, 32]
    log_ratios = np.log2(np.array(ratios, dtype=float))
    norm = mcolors.Normalize(vmin=log_ratios.min(), vmax=log_ratios.max())
    ratio_colors = {r: CMAP(norm(np.log2(r))) for r in ratios}

    dfi_full, _, _ = load_protein_data(protein_key)
    if dfi_full is None:
        print(f"Vis data missing for {protein_key}.")
        return
    tm_regs = GPCR_TM_REGIONS[protein_key]
    x_min, x_max = tm_regs[0][0], tm_regs[-1][1]
    sweep_trim = sweep_data[(sweep_data["ResI"] >= x_min) & (sweep_data["ResI"] <= x_max)].copy()
    sweep_trim["H"] = np.log(sweep_trim["DFI_Z"]) - np.log(sweep_trim["DFI_XY"])

    fig = plt.figure(figsize=(9.0, 6.5))
    gs = gridspec.GridSpec(4, 1, hspace=0.03,
                           left=0.15, right=0.72,
                           top=0.92, bottom=0.08)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    ax_tot, ax_z, ax_xy, ax_h = axes

    metrics_top = [
        ("DFI_total" if "DFI_total" in sweep_data.columns else "DFI_membrane", r"DFI total ($\times 10^{-3}$)"),
        ("DFI_Z", r"DFI z ($\times 10^{-3}$)"),
        ("DFI_XY", r"DFI xy ($\times 10^{-3}$)")
    ]
    for i, (ax, (col, label)) in enumerate(zip([ax_tot, ax_z, ax_xy], metrics_top)):
        _draw_tm_shading_and_labels(ax, tm_regs, x_min, x_max, show_labels=(i == 0))
        for ratio in ratios:
            sub = sweep_trim[sweep_trim["ratio"] == ratio]
            y_vals = sub[col].rolling(window=5, center=True, min_periods=1).mean() * 1000.0
            ax.plot(sub["ResI"], y_vals, color=ratio_colors[ratio], ls="-", lw=0.8, alpha=0.9, zorder=3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel(label, fontsize=12.5)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)
        apply_black_border(ax)

    _draw_tm_shading_and_labels(ax_h, tm_regs, x_min, x_max, show_labels=False)
    for ratio in ratios:
        sub = sweep_trim[sweep_trim["ratio"] == ratio]
        y_vals = sub["H"].rolling(window=5, center=True, min_periods=1).mean()
        ax_h.plot(sub["ResI"], y_vals, color=ratio_colors[ratio], ls="-", lw=1.0, alpha=0.9, zorder=3)
    ax_h.set_xlim(x_min, x_max)
    ax_h.set_ylabel("rdDFI", labelpad=4, fontsize=12.5)
    ax_h.set_xlabel("Residue index", fontsize=16, fontweight='normal')
    apply_black_border(ax_h)

    cax = ax_h.inset_axes([0.02, 0.85, 0.32, 0.12])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_ticks(log_ratios)
    cbar.set_ticklabels([str(int(r)) for r in ratios])

    zoom_x_lim = (198, 226)
    add_zoom_inset(ax_z, sweep_trim, ratios, ratio_colors, "DFI_Z", zoom_x_lim,
                   arrow_color="#3B82F6", arrow_xy=(0.20, 0.65), arrow_xytext=(0.20, 0.90))
    add_zoom_inset(ax_xy, sweep_trim, ratios, ratio_colors, "DFI_XY", zoom_x_lim,
                   arrow_color="#D97706", arrow_xy=(0.20, 0.90), arrow_xytext=(0.20, 0.65))
    add_zoom_inset(ax_h, sweep_trim, ratios, ratio_colors, "H", zoom_x_lim,
                   arrow_color="#FF0000", arrow_xy=(0.20, 0.10), arrow_xytext=(0.20, 0.35))

    out_file = os.path.join(OUT_DIR, "Figure_4.png")
    fig.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 4: {out_file}")

# =====================================================================
# 4. Figure 5
# =====================================================================
def plot_figure5(protein_key="acm1", summary_csv_name="Prediction_models_hit_summary_2.csv"):
    summary_csv = os.path.join(SCRIPT_DIR, summary_csv_name)
    print("Generating Figure 5...")
    fig = plt.figure(figsize=(8.0, 9.5))
    gs_main = gridspec.GridSpec(3, 1, height_ratios=[0.75, 0.85, 1.25], hspace=0.45)

    # ---------------------------------------------------------
    # Layer 1: Venn diagram (rdDFI label, smaller font, inside top of circles)
    # ---------------------------------------------------------
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0],
                                               width_ratios=[1.2, 0.8], wspace=0.0)
    ax_venn = fig.add_subplot(gs_row1[0])
    ax_dummy = fig.add_subplot(gs_row1[1])
    ax_dummy.axis('off')

    ax_venn.set_box_aspect(35/50)
    ax_venn.axis('off')

    dfi_only_count = 0
    h_only_count = 0
    both_count = 0

    for prot in GPCR_TM_REGIONS.keys():
        dfi_f, _, _ = load_protein_data(prot)
        if dfi_f is not None:
            tm_r = GPCR_TM_REGIONS[prot]
            tm_res = set(r for a, b in tm_r for r in range(a, b + 1))
            dfi_t = dfi_f[dfi_f["ResI"].isin(tm_res)].copy()
            if len(dfi_t) > 0:
                s_dfi = dfi_t['DFI'].astype(float)
                s_h = dfi_t['H'].astype(float)
                z_dfi = (s_dfi - s_dfi.mean()) / s_dfi.std(ddof=0) if s_dfi.std(ddof=0) > 0 else s_dfi * 0.0
                z_h = (s_h - s_h.mean()) / s_h.std(ddof=0) if s_h.std(ddof=0) > 0 else s_h * 0.0
                dfi_sel = set(dfi_t[z_dfi < Z_THRESH]['ResI'].tolist())
                h_sel   = set(dfi_t[z_h < Z_THRESH]['ResI'].tolist())
                both_count     += len(dfi_sel & h_sel)
                dfi_only_count += len(dfi_sel - h_sel)
                h_only_count   += len(h_sel - dfi_sel)

    total_dfi = dfi_only_count + both_count
    total_h = h_only_count + both_count

    # ---------- 圆面积大幅缩小（系数 0.05 + 偏移 0.02） ----------
    r_dfi = 0.20 * math.sqrt(max(total_dfi, 1) / max(total_dfi + total_h, 1)) + 0.08
    r_h   = 0.20 * math.sqrt(max(total_h, 1) / max(total_dfi + total_h, 1)) + 0.08

    c_left  = METRIC_STYLE["DFI"]["color"]
    c_right = METRIC_STYLE["H"]["color"]

    circle_dfi = patches.Circle((0.35, 0.5), r_dfi, color=c_left, alpha=0.5,
                                transform=ax_venn.transAxes, edgecolor="grey", linewidth=1.5, clip_on=False)
    circle_h   = patches.Circle((0.65, 0.5), r_h, color=c_right, alpha=0.5,
                                transform=ax_venn.transAxes, edgecolor="grey", linewidth=1.5, clip_on=False)
    ax_venn.add_patch(circle_dfi)
    ax_venn.add_patch(circle_h)

    # ---------- 标签（无多余空格，居中准确） ----------
    ax_venn.text(0.35, 0.62, "     DFI", ha='center', va='center',
                 fontsize=11, fontweight='bold', color='black', transform=ax_venn.transAxes, zorder=5)
    ax_venn.text(0.65, 0.62, "rdDFI     ", ha='center', va='center',
                 fontsize=11, fontweight='bold', color='black', transform=ax_venn.transAxes, zorder=5)

    ax_venn.text(0.23, 0.48, f"  {dfi_only_count}", ha='center', va='center',
                 fontsize=12, fontweight='bold', color='black', transform=ax_venn.transAxes, zorder=5)
    ax_venn.text(0.77, 0.48, f" {h_only_count}", ha='center', va='center',
                 fontsize=12, fontweight='bold', color='black', transform=ax_venn.transAxes, zorder=5)
    ax_venn.text(0.50, 0.48, f"{both_count}  ", ha='center', va='center',
                 fontsize=12, fontweight='bold', color='black', transform=ax_venn.transAxes, zorder=5)

    # ---------------------------------------------------------
    # Layer 2: Overview Bar Chart
    # ---------------------------------------------------------
    ax_over = fig.add_subplot(gs_main[1])
    if os.path.exists(summary_csv):
        df = pd.read_csv(summary_csv)
        if 'HitRate' not in df.columns and 'Precision' in df.columns:
            df['HitRate'] = df['Precision'] * 100
        df['Metric_Label'] = df['Metric'].map(lambda x: METRIC_STYLE.get(x, {}).get("label", x))
        df = df[df["Metric"].isin(RUG_METRICS)]
        proteins = sorted(df["Protein"].unique())
        x = np.arange(len(proteins))
        width = 0.85 / len(RUG_METRICS)
        center_offset = (len(RUG_METRICS) - 1) / 2.0

        all_vals = []
        for i, metric in enumerate(RUG_METRICS):
            vals = []
            for p in proteins:
                val = df[(df["Protein"] == p) & (df["Metric"] == metric)]["HitRate"]
                vals.append(val.values[0] if len(val) > 0 else 0)
            all_vals.append(vals)
            style = METRIC_STYLE[metric]
            ax_over.bar(x + (i - center_offset) * width, vals, width=width,
                        color=style["color"], edgecolor="none", alpha=0.95, label=style["label"])

        h_idx = RUG_METRICS.index("H")
        for j, p in enumerate(proteins):
            rates = [all_vals[k][j] for k in range(len(RUG_METRICS))]
            max_val = max(rates)
            if rates[h_idx] == max_val and rates[h_idx] > 0:
                x_pos = x[j] + (h_idx - center_offset) * width
                y_pos = rates[h_idx]
                ax_over.plot(x_pos, y_pos + 3.0, marker='*', color='#FFD700',
                             markersize=13, linestyle='None')

        ax_over.set_xticks(x)
        ax_over.set_xticklabels([p.upper() for p in proteins], rotation=45, ha="right")
        
        for tick in ax_over.xaxis.get_major_ticks():
            if tick.label1.get_text() == "ACM1":
                tick.label1.set_color("#FF0000")
                tick.label1.set_fontweight("bold")
                tick.tick1line.set_markeredgecolor("#FF0000")
                tick.tick1line.set_color("#FF0000")
                tick.tick1line.set_markeredgewidth(1.5)
                
        ax_over.set_ylabel("Hit Rate (%)")
        ax_over.legend(loc="upper left", ncol=3, frameon=True, edgecolor="black", facecolor="white", fontsize=11)
        apply_black_border(ax_over)

    # ---------------------------------------------------------
    # Layer 3: acm1 Vis
    # ---------------------------------------------------------
    gs_row3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[2], height_ratios=[1.3, 0.35], hspace=0.08)

    dfi_full, mutated_positions_full, pos_mut_count = load_protein_data(protein_key)
    if dfi_full is not None:
        tm_regs = GPCR_TM_REGIONS[protein_key]
        x_min, x_max = tm_regs[0][0], tm_regs[-1][1]

        dfi_plot = dfi_full[(dfi_full["ResI"] >= x_min - 2) & (dfi_full["ResI"] <= x_max + 2)].copy()
        tm_residues = set(r for a, b in tm_regs for r in range(a, b + 1))
        dfi_tm = dfi_full[dfi_full["ResI"].isin(tm_residues)].copy()
        mutated_positions_tm = mutated_positions_full & tm_residues

        results = {}
        for m in RUG_METRICS:
            results[m] = evaluate_selection(dfi_tm, m, mutated_positions_tm)

        ax_main = fig.add_subplot(gs_row3[0, 0])
        ax_rug = fig.add_subplot(gs_row3[1, 0], sharex=ax_main)
        ax_line = ax_main.twinx()

        _draw_tm_shading_and_labels(ax_main, tm_regs, x_min, x_max, show_labels=True)
        _draw_tm_shading_and_labels(ax_rug, tm_regs, x_min, x_max, show_labels=False)

        positions = sorted(pos_mut_count.keys())
        bar_heights = np.array([pos_mut_count[p] for p in positions])
        bar_colors = [PALETTE["mut_tm"] if p in tm_residues else PALETTE["mut_nontm"] for p in positions]
        max_bar = max(bar_heights) if len(bar_heights) else 1
        ax_main.bar(positions, bar_heights, width=0.78, color=bar_colors, edgecolor="none", zorder=2)
        ax_main.set_xlim(x_min - 2, x_max + 2)
        ax_main.set_ylim(0, max_bar * 1.25 + 1)
        ax_main.set_ylabel("Mutation count")
        ax_main.yaxis.tick_right()
        ax_main.yaxis.set_label_position("right")
        ax_main.set_title(f"{protein_key.upper()}", pad=6, color="#FF0000", fontweight="bold")
        plt.setp(ax_main.get_xticklabels(), visible=False)

        for metric, col in [("DFI", "DFI"), ("DFI_membrane", "DFI_membrane"), ("H", "H")]:
            tm_s = dfi_tm[col].astype(float)
            tm_mean = tm_s.mean()
            tm_std = tm_s.std(ddof=0)
            
            plot_s = dfi_plot[col].astype(float)
            if tm_std > 0:
                plot_z = (plot_s - tm_mean) / tm_std
            else:
                plot_z = plot_s * 0.0
                
            dfi_plot[f"{col}_norm"] = plot_z.rolling(window=7, center=True, min_periods=1).mean()
            
            style = METRIC_STYLE[metric]
            ax_line.plot(dfi_plot["ResI"], dfi_plot[f"{col}_norm"],
                         color=style["color"], linestyle=style["ls"], linewidth=1.5, alpha=0.9, zorder=4)

        ax_line.set_ylabel(r"$Z_{TM}$ score")
        ax_line.yaxis.tick_left()
        ax_line.yaxis.set_label_position("left")
        ax_line.set_yticks([-4, 0, 4, 8, 12, 16])
        ax_line.tick_params(axis='y', labelsize=11)
        
        ax_line.axhline(y=-1, color="black", linestyle="--", linewidth=1.2, alpha=0.6, zorder=1)
        tm1_center = (tm_regs[0][0] + tm_regs[0][1]) / 2.0
        ax_line.text(tm1_center, -1.3, r"$Z_{TM}=-1$", color="black", 
                     fontsize=6.5, ha="center", va="top", zorder=5)

        handles = [mlines.Line2D([], [], color=v["color"], label=v["label"], lw=2) for k, v in METRIC_STYLE.items()]
        handles.append(mlines.Line2D([], [], color="none", marker="^",
                                     markerfacecolor="#CBD5E1", markeredgecolor="#CBD5E1",
                                     markersize=5, linestyle="None", label="Uncaptured"))
        ax_line.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.89), ncol=2,
                       frameon=True, edgecolor="black", facecolor="white", framealpha=0.9, fontsize=11)

        ax_rug.set_ylim(0, len(RUG_METRICS))
        ax_rug.set_yticks([i + 0.5 for i in range(len(RUG_METRICS))])
        ax_rug.set_yticklabels([])
        ax_rug.set_ylabel("Hit\nPositions", labelpad=15, rotation=90, va="center")

        for row_i, metric in enumerate(RUG_METRICS):
            y_center = row_i + 0.5
            style = METRIC_STYLE[metric]
            res = results[metric]
            for pos in res["misses"]:
                ax_rug.plot(pos, y_center, marker="^", color="#CBD5E1", markersize=2.5, linestyle="None", zorder=2)
            for pos in res["hits"]:
                ax_rug.plot(pos, y_center, marker="^", color=style["color"], markersize=4, linestyle="None", zorder=4)

        ax_rug.set_xlabel("Residue index")
        ax_rug.xaxis.set_major_locator(MultipleLocator(50))
        apply_black_border(ax_main)
        apply_black_border(ax_line)
        apply_black_border(ax_rug)

    out_file = os.path.join(OUT_DIR, "Figure_5.png")
    fig.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 5: {out_file}")

if __name__ == "__main__":
    plot_figure4()
    plot_figure5()
