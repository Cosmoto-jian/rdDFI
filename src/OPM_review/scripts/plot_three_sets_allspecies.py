#!/usr/bin/env python3
"""Three 3x2 figures for the all-species selected sets (OPM, isolated chain imANM, K=1000).
Rows: ab = four alpha-helical channels; cd = four beta-barrels; ef = four classes.
Fig1: φ vs r_ij (s=16), inset Length–Rg in right panels; Fig2: φ vs ρ (s=16); Fig3: ρ* vs s.
Legend 2x2 in upper right of the left panel of each row; "s = 16" level with it at upper left (Fig1/2)."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator, FormatStrFormatter
from matplotlib.lines import Line2D

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "figures"; OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4; S_MAIN = 16; S_VALUES = [1, 2, 4, 8, 16, 32]
from matplotlib.colors import to_rgb
def ramp(base):  # light -> dark, ordered by length
    b = np.array(to_rgb(base)); w, k = np.ones(3), np.zeros(3)
    return [tuple(b + (w - b) * 0.45), tuple(b + (w - b) * 0.2), tuple(b), tuple(b * 0.62)]
CLASS = ["#3C5488", "#00A087", "#E64B35", "#8E5EA2"]
ROWCS = [ramp("#00A087"), ramp("#8E5EA2"), CLASS]
ROWTITLE = ["α-helical channels", "β-barrels", "Four classes"]
MK = ["o", "s", "^", "D"]; MINFRAC = 0.05
XMAX_R = [100, 80, 50]; HEAD1L = [0.05, 0.05, 0.05]; XMAX_RHO = 3.5; HEAD3 = [0.45, 0.12, 0.12]; INSET_POS = [[0.44, 0.60, 0.32, 0.30], [0.38, 0.47, 0.36, 0.36], [0.38, 0.47, 0.36, 0.36]]
MS = {marker: 4.0 for marker in MK}
R10 = DATA / "imANM_results"
# (tag, legend label, Rg Å)  — ab/cd ordered by length; ef ordered by class (fixed class colours)
SETS = [
    [("7sk1_A", "TWIK1", 24.1), ("5u1x_A", "P2X7", 29.2), ("6bwj_A", "TRPV2", 34.3), ("6bpq_A", "TRPM8", 40.4)],
    [("4y25_A", "PgaA", 20.1), ("2hdi_A", "Cir", 24.7), ("4zgv_A", "FusA", 29.1), ("6e14_D", "FimD", 34.9)],
    [("2xfn_A", "Bitopic MAO-B", 23.0), ("6o84_A", "Channel OTOP3", 22.4), ("8cv2_A", "Transporter EAAT3", 22.2), ("2mpr_A", "β-barrel LamB", 23.2)],
]
RES = [{p: json.load(open(R10 / f"{p}.json")) for p, _, _ in row} for row in SETS]

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"], "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold", "legend.frameon": True, "savefig.dpi": 600,
    "font.size": 8, "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 6,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round"})

CS = CLASS
def line(ax, x, y, i, every=1, row=0):
    c = ROWCS[row][i]
    ax.plot(x, y, "-", color=c, lw=1.4, marker=MK[i], ms=MS[MK[i]], mfc=c, mec="black", mew=0.3, zorder=3, markevery=every)

def curve(R, p, d, norm):
    g = R[p]["s"][str(S_MAIN)][d]; r = np.array(g["r"]); f = np.array(g["phi"]); c = np.array(g["cnt"])
    ok = c >= MINFRAC * c.max(); im = int(np.argmax(c)); bad = np.where(~ok[im:])[0]
    n = im + bad[0] if len(bad) else len(r); r, f = r[:n], f[:n]   # truncate sparse long-range tail only
    if norm: r = r / (R[p]["L_xy"] if d == "xy" else R[p]["L_z"])
    return r, f

def nice_y(ax, lo, hi, head, step=None, nb=5):
    """y-limits hug the data (5% below, `head` above); ticks are the nice values inside the limits."""
    span = hi - lo; y0, y1 = lo - 0.05 * span, hi + max(head, 0.05) * span
    major_step = 0.2
    locator = MultipleLocator(major_step)
    while np.count_nonzero((locator.tick_values(y0, y1) >= y0) & (locator.tick_values(y0, y1) <= y1)) < 3:
        y0 -= major_step / 2; y1 += major_step / 2
    ax.set_ylim(y0, y1)
    ax.yaxis.set_major_locator(MultipleLocator(major_step))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

def rowtitle(ax, row):
    ax.text(-0.50, 0.5, ROWTITLE[row], transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=11, fontweight="bold", color="0.25")

def headroom(ax, f):
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + f * (hi - lo))

FS = dict(tick=8, lab=10, ann=6, let=14)
def set_fs(**k):
    FS.update(k); plt.rcParams.update({"xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "axes.labelsize": FS["lab"]})

def ylab(ax, t, pad=5): ax.set_ylabel(t, rotation=0, ha="right", va="center", labelpad=pad)

def tag_legend(ax, row, s_label=True, top=0.975, margin=0.025, auto=False, gap=0.04):
    """Combined type heading and curve labels in one framed legend."""
    fs = FS["ann"]; fig = ax.figure; r = fig.canvas.get_renderer()
    h = [Line2D([], [], color=ROWCS[row][i], lw=1.4, marker=MK[i], ms=MS[MK[i]], mec="black", mew=0.3, label=lab) for i, (_, lab, _) in enumerate(SETS[row])]
    txt = TAG[row] + (f" ($s$ = {S_MAIN})" if s_label else "")
    lg = ax.legend(handles=h, title=txt, loc="upper right", bbox_to_anchor=(1 - margin, top), ncol=1,
                   fontsize=fs, title_fontsize=fs, handlelength=1.3, handletextpad=0.35,
                   labelspacing=0.25, borderpad=0.35, borderaxespad=0, frameon=True, fancybox=True, shadow=True)
    f = lg.get_frame(); f.set_edgecolor("0.65"); f.set_linewidth(0.6); f.set_facecolor("white"); f.set_alpha(1); lg.set_zorder(5)
    lg.get_title().set_fontweight("bold")
    if auto:
        fig.canvas.draw(); A = ax.get_window_extent(r); L = lg.get_window_extent(r)
        bottom = (L.y0 - A.y0) / A.height
        x0 = (L.x0 - A.x0) / A.width
        for _ in range(60):
            tr = ax.transData + ax.transAxes.inverted(); hit = 0
            for ln in ax.get_lines():
                d = np.column_stack(ln.get_data())
                if len(d) < 2 or np.ptp(d[:, 0]) == 0: continue
                xy = tr.transform(d); xs = np.linspace(x0 - 0.02, 1, 200)
                m = (xs >= xy[:, 0].min()) & (xs <= xy[:, 0].max())
                if m.any(): hit = max(hit, np.interp(xs[m], xy[:, 0], xy[:, 1]).max())
            if hit <= bottom - gap: break
            y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1 + 0.03 * (y1 - y0))

TAG = ["α-helical channels", "β-barrels", "Four classes"]
BOX = dict(boxstyle="round,pad=0.35,rounding_size=0.25", fc="white", ec="0.45", lw=0.6)
def tag(ax, row, s_label=True, loc="right", fs=8.5):
    t = TAG[row] + (f" ($s$ = {S_MAIN})" if s_label else "")
    x, ha = (0.97, "right") if loc == "right" else (0.04, "left")
    ax.text(x, 0.96, t, transform=ax.transAxes, ha=ha, va="top", fontsize=fs, color="0.15", bbox=BOX, zorder=5)

def legend(ax, row, loc="upper right", anchor=(1.0, 1.0), fs=6.9, frame=False):
    h = [Line2D([], [], color=ROWCS[row][i], lw=1.4, marker=MK[i], ms=MS[MK[i]], mec="black", mew=0.3, label=lab) for i, (_, lab, _) in enumerate(SETS[row])]
    lg = ax.legend(handles=h, loc=loc, bbox_to_anchor=anchor, ncol=1 if row == 0 else 2, fontsize=fs, handlelength=1.3,
              handletextpad=0.3, columnspacing=0.6, labelspacing=0.3, borderpad=0.4, borderaxespad=0.3, frameon=True, fancybox=True)
    f = lg.get_frame(); f.set_edgecolor("0.45" if frame else "none"); f.set_linewidth(0.6); f.set_facecolor("white"); f.set_alpha(1); lg.set_zorder(5)

def inset(ax, row):
    R = RES[row]; N = np.array([R[p]["N"] for p, _, _ in SETS[row]]); G = np.array([g for _, _, g in SETS[row]])
    ins = ax.inset_axes(INSET_POS[row]); o = np.argsort(N)
    ins.plot(N[o], G[o], ls=(0, (1.2, 1.4)), color="0.45", lw=0.9, zorder=1)
    for i in range(4):
        ins.plot(N[i], G[i], marker=MK[i], ms=MS[MK[i]], mfc=ROWCS[row][i], mec="black", mew=0.3, ls="none", zorder=3)
    ins.tick_params(labelsize=FS["tick"], direction="out", length=2.5, width=0.5, pad=1.2)
    ins.tick_params(which="minor", direction="out", length=1.5, width=0.4)
    for sp in ins.spines.values(): sp.set_linewidth(0.5); sp.set_visible(True)
    ins.set_xlabel("N", fontsize=FS["lab"], labelpad=1)
    ins.yaxis.tick_right(); ins.yaxis.set_label_position("right")
    ins.set_ylabel(r"$R_g$ (Å)", fontsize=FS["lab"], labelpad=1)
    ins.xaxis.set_minor_locator(AutoMinorLocator(5)); ins.yaxis.set_minor_locator(AutoMinorLocator(5))
    ins.xaxis.set_major_locator(MaxNLocator(2, integer=True)); ins.yaxis.set_major_locator(MultipleLocator(10))
    ins.xaxis.set_major_formatter(FormatStrFormatter("%.0f")); ins.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ins.set_ylim(10, 45)
    ins.margins(x=0.15, y=0.18); ins.patch.set_alpha(0)

PWX, PWY = 39, 28; ML, MGAP, MR, MB, MT, MV = 27, 16, 6, 22, 10, 20   # mm; panel ratio matches the reference axes
def new_fig():
    W = ML + 2 * PWX + MGAP + MR; H = MB + 3 * PWY + 2 * MV + MT
    fig = plt.figure(figsize=(W * MM, H * MM)); axes = np.empty((3, 2), dtype=object)
    for r in range(3):
        for c in range(2):
            x = ML + c * (PWX + MGAP); y = MB + (2 - r) * (PWY + MV)
            axes[r, c] = fig.add_axes([x / W, y / H, PWX / W, PWY / H])
    return fig, axes

def finish(fig, axes, name):
    fig.align_ylabels(axes[:, 0]); fig.align_ylabels(axes[:, 1])
    for a, t in zip(axes.flat, "abcdef"):
        a.text(-0.36, 1.02, t, transform=a.transAxes, fontsize=FS["let"], va="bottom")
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def phi_fig(norm, name):
    fig, axes = new_fig()
    for row in range(3):
        R = RES[row]; ys = []
        for col, d in enumerate(["xy", "z"]):
            ax = axes[row, col]
            for i, (p, _, _) in enumerate(SETS[row]):
                x, y = curve(R, p, d, norm)
                k = x <= (XMAX_RHO if norm else XMAX_R[row]); x, y = x[k], y[k]
                line(ax, x, y, i, every=2, row=row); ys.append(y)
            ax.axhline(0, color="0.55", lw=0.5, ls=(0, (3, 2)), zorder=0)
            ylab(ax, r"$\phi_{xy}$" if d == "xy" else r"$\phi_{z}$", pad=5 if col == 0 else 3)
            if norm:
                ax.set_xlabel(r"$\rho_{xy}$" if d == "xy" else r"$\rho_{z}$", labelpad=2)
                ax.set_xlim(0, XMAX_RHO * 1.02); ax.xaxis.set_major_locator(MultipleLocator(1)); ax.xaxis.set_minor_locator(MultipleLocator(0.2))
            else:
                ax.set_xlabel(r"$r_{ij}$ (Å)", labelpad=2)
                ax.set_xlim(0, XMAX_R[row] * 1.02); ax.xaxis.set_major_locator(MultipleLocator(20)); ax.xaxis.set_minor_locator(MultipleLocator(4))
        for col in range(2):
            ax = axes[row, col]; yy = np.concatenate(ys[col * 4:(col + 1) * 4])
            nice_y(ax, yy.min(), yy.max(), 0.05 if col == 1 else (0.05 if norm else HEAD1L[row]), nb=6)
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.tick_params(axis="y", which="minor", direction="out", length=1.5, width=0.4)
        if not norm:
            inset(axes[row, 0], row); tag_legend(axes[row, 1], row, auto=True)
        else:
            tag_legend(axes[row, 1], row, auto=True)
    finish(fig, axes, name)

def rho_fig(name):
    fig, axes = new_fig()
    for row in range(3):
        R = RES[row]
        for col, c in enumerate(["rho_xy", "rho_z"]):
            ax = axes[row, col]
            for i, (p, _, _) in enumerate(SETS[row]):
                line(ax, S_VALUES, [R[p]["s"][str(s)][c] for s in S_VALUES], i, row=row)
            ax.set_xscale("log", base=2); ax.set_xticks(S_VALUES); ax.set_xticklabels([str(s) for s in S_VALUES]); ax.minorticks_off()
            v = [R[p]["s"][str(s)][c] for p, _, _ in SETS[row] for s in S_VALUES]
            nice_y(ax, min(v), max(v), 0.05, nb=4)
            ax.axvline(S_MAIN, color="0.6", lw=0.5, ls=(0, (2, 2)), zorder=0)
            ax.set_xlabel(r"$s$", labelpad=2)
            ylab(ax, r"$\rho^{*}_{xy}$" if col == 0 else r"$\rho^{*}_{z}$", pad=5 if col == 0 else 3)
        tag_legend(axes[row, 1], row, s_label=False, auto=True)
    finish(fig, axes, name)

set_fs(tick=8, lab=10, ann=6, let=14); phi_fig(False, "Fig1_phi_vs_rij_allspecies_sets")
set_fs(tick=8, lab=10, ann=6, let=14); phi_fig(True, "Fig2_phi_vs_rho_allspecies_sets")
set_fs(tick=8, lab=10, ann=6, let=14); rho_fig("Fig3_rhostar_vs_s_allspecies_sets")
print("ok")
