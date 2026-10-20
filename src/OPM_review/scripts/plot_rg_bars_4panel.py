#!/usr/bin/env python3
"""Four-panel bar chart: Rg (mean ± SD, points = single chains) by length bin, one panel per OPM class.
Pool = OPM all-species representatives, 250–850 Cα (data/representatives_250_850.csv)."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
MM = 1 / 25.4; DATA = Path(__file__).resolve().parents[1] / "data"; OUT = Path(__file__).resolve().parents[1] / "figures"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial"], "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic", "axes.linewidth": 0.8, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8, "axes.labelsize": 14, "xtick.labelsize": 9, "ytick.labelsize": 9, "savefig.dpi": 600})
d = pd.read_csv(DATA / "representatives_250_850.csv")
CL = [("A_bitopic", "Bitopic", "#3C5488"), ("B_receptor_channel", "Receptor/channel", "#00A087"),
      ("C_transporter", "Transporter", "#E64B35"), ("D_beta_barrel", "β-barrel", "#8E5EA2")]
B = ["250–400", "400–550", "550–700", "700–850"]; rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 2, figsize=(150 * MM, 118 * MM), sharey=True)
rows = []
for ax, (c, lab, col), let in zip(axes.flat, CL, "abcd"):
    sub = d[d.cls == c]
    for i, b in enumerate(B):
        v = sub[sub.bin == b].Rg.values
        if len(v) == 0: continue
        m = v.mean(); sd = v.std(ddof=1) if len(v) > 1 else 0.0
        rows.append(dict(cls=lab, bin=b, n=len(v), mean=m, sd=sd, median=np.median(v)))
        ax.bar(i, m, 0.62, color=col, alpha=0.85, lw=0, zorder=2)
        if len(v) > 1: ax.errorbar(i, m, yerr=sd, fmt="none", ecolor="0.15", elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=2.6, color="white", edgecolor="0.25", linewidth=0.35, zorder=3)
        ax.text(i, 2.5, f"{len(v)}", ha="center", va="bottom", fontsize=8.5, color="white", fontweight="bold", zorder=5)
    ax.set_xticks(range(4)); ax.set_xticklabels(B); ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20)); ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.text(0.03, 0.97, f"{lab}  (n = {len(sub)})", transform=ax.transAxes, ha="left", va="top", fontsize=10, color=col, fontweight="bold")
    ax.text(-0.24, 1.02, let, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
for ax in axes[:, 0]: ax.set_ylabel(r"$R_g$ (Å)", rotation=0, ha="right", va="center", labelpad=8)
for ax in axes[1]: ax.set_xlabel(r"Length ($N_{\mathrm{C\alpha}}$)", labelpad=3)
fig.subplots_adjust(left=0.13, right=0.99, bottom=0.11, top=0.95, hspace=0.45, wspace=0.12)
fig.text(0.56, -0.035, f"OPM, all species, one chain per protein, 250–850 Cα; total n = {len(d)} proteins. Bars: mean ± SD.",
         ha="center", va="bottom", fontsize=8, color="0.4")
for e in ["pdf", "svg", "png"]: fig.savefig(OUT / f"Fig4_Rg_by_length_4classes_allspecies.{e}", bbox_inches="tight", pad_inches=0.03)
pd.DataFrame(rows).round(2).to_csv(DATA / "Rg_by_length_4classes.csv", index=False)
print(pd.DataFrame(rows).round(1).to_string())
