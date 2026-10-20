#!/usr/bin/env python3
"""Grouped bar chart: Rg (mean ± SD) of OPM all-species single chains by length bin and class, rdDFI Nature style."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
DATA = Path(__file__).resolve().parents[1] / "data"; MM=1/25.4
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Liberation Sans","Arial"],"axes.spines.top":False,"axes.spines.right":False,
 "xtick.direction":"out","ytick.direction":"out","pdf.fonttype":42,"svg.fonttype":"none","mathtext.fontset":"custom","mathtext.rm":"Liberation Sans",
 "mathtext.it":"Liberation Sans:italic","axes.linewidth":0.8,"xtick.major.width":0.8,"ytick.major.width":0.8,"axes.labelsize":14,"xtick.labelsize":9,"ytick.labelsize":9,"savefig.dpi":600})
d=pd.read_csv(DATA / 'representatives_250_850.csv')
CL=[('A_bitopic','Bitopic','#3C5488'),('B_receptor_channel','Receptor/channel','#00A087'),('C_transporter','Transporter','#E64B35'),('D_beta_barrel','β-barrel','#8E5EA2')]
B=['250–400','400–550','550–700','700–850']; w=0.19; rng=np.random.default_rng(0)
fig,ax=plt.subplots(figsize=(130*MM,80*MM))
for i,b in enumerate(B):
    present=[(c,lab,col) for c,lab,col in CL if ((d.cls==c)&(d.bin==b)).any()]
    for k,(c,lab,col) in enumerate(present):
        v=d[(d.cls==c)&(d.bin==b)].Rg.values; x=i+(k-(len(present)-1)/2)*w
        m=v.mean(); sd=v.std(ddof=1) if len(v)>1 else 0
        ax.bar(x,m,w*0.92,color=col,alpha=0.85,lw=0,zorder=2)
        if len(v)>1: ax.errorbar(x,m,yerr=sd,fmt='none',ecolor='0.15',elinewidth=0.9,capsize=2.0,capthick=0.9,zorder=4)
        ax.scatter(x+rng.uniform(-w*0.28,w*0.28,len(v)),v,s=2.2,color='white',edgecolor='0.2',linewidth=0.35,zorder=3)
        ax.text(x,1.5,f'{len(v)}',ha='center',va='bottom',fontsize=7,color='white',fontweight='bold',zorder=5)
ax.set_xticks(range(4)); ax.set_xticklabels(B); ax.set_xlim(-0.55,3.55)
ax.set_xlabel(r'Length ($N_{\mathrm{C\alpha}}$)',labelpad=2); ax.set_ylabel(r'$R_g$ (Å)',rotation=0,ha='right',va='center',labelpad=8)
ax.set_ylim(0,90)
ax.legend(handles=[Patch(color=col,alpha=0.85,label=lab) for _,lab,col in CL],loc='lower center',ncol=4,bbox_to_anchor=(0.5,1.01),borderaxespad=0,fontsize=10,frameon=False,handlelength=1.0,handletextpad=0.4,columnspacing=0.9)
t=d.groupby('cls').Rg.agg(['count','mean','std'])
h=d.groupby('cls').human.sum()
note=f'n = {len(d)} OPM proteins (all species, 250–850 Cα). Bars: mean ± SD.'
ax.text(0.5,-0.33,note,transform=ax.transAxes,ha='center',va='top',fontsize=7.5,color='0.2')
fig.subplots_adjust(left=0.17,right=0.99,bottom=0.25,top=0.87)
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)
for e in ['pdf','svg','png']: fig.savefig(OUT / f'Fig4_Rg_by_length_grouped_allspecies.{e}',bbox_inches='tight',pad_inches=0.02)
print('ok')
