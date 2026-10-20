#!/usr/bin/env python3
"""Usage: python plot_xi_scatter_L_Rg.py 16   (or 1)
OPM pool (928, s=16). a,b: xi_alpha vs rho*_alpha, colour = N; c,d: xi_alpha vs R_g with linear fits + Pearson r. Marker: alpha filled circle, beta open triangle."""
import json,os,csv,math,numpy as np,matplotlib; matplotlib.use("Agg")
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
FD=os.path.expanduser('~/.fonts')
if os.path.isdir(FD):
    for f in os.listdir(FD): fm.fontManager.addfont(os.path.join(FD,f))
import sys; SS=sys.argv[1] if len(sys.argv)>1 else "16"   # "16" or "1"
MM=1/25.4
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Liberation Sans"],"axes.spines.top":False,"axes.spines.right":False,
 "xtick.direction":"out","ytick.direction":"out","pdf.fonttype":42,"svg.fonttype":"none","mathtext.fontset":"custom","mathtext.rm":"Liberation Sans",
 "mathtext.it":"Liberation Sans:italic","axes.linewidth":0.8,"xtick.major.width":0.8,"ytick.major.width":0.8,"axes.labelsize":14,"xtick.labelsize":9,"ytick.labelsize":9,"savefig.dpi":600})
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'opm_xi_L_Rg_s1_s16.csv'
OUT=ROOT/'figures'
OUT.mkdir(parents=True,exist_ok=True)
D=[]
for r in csv.DictReader(open(DATA)):
    D.append(dict(g='b' if r['group']=='beta-barrel' else 'a',N=int(r['N']),Rg=float(r['Rg']),Lxy=float(r['L_xy']),Lz=float(r['L_z']),
                  xixy=float(r[f'xi_xy_s{SS}']),xiz=float(r[f'xi_z_s{SS}'])))
na=sum(d['g']=='a' for d in D); nb=len(D)-na
cmap=LinearSegmentedColormap.from_list('n',['#C9D6EA','#7E9CC6','#3C5488','#1B2541']); norm=Normalize(250,850)
BINS=[(250,400),(400,550),(550,700),(700,851)]
MKa=dict(marker='o',s=8,lw=0.25,edgecolors='white',alpha=0.8)
MKb=dict(marker='^',s=11,lw=0.6,facecolors='none',alpha=0.9)
fig,axs=plt.subplots(2,2,figsize=(150*MM,118*MM)); axs=axs.ravel()
stats={}
for k,(xk,yk,xl,yl) in enumerate([('Lxy','xixy',r'$L_{xy}$ (Å)',r'$\xi_{xy}$ (Å)'),('Lz','xiz',r'$L_z$ (Å)',r'$\xi_z$ (Å)'),
                                   ('Rg','xixy',r'$R_g$ (Å)',r'$\xi_{xy}$ (Å)'),('Rg','xiz',r'$R_g$ (Å)',r'$\xi_z$ (Å)')]):
    ax=axs[k]
    order=sorted(D,key=lambda d:d['N'])
    A=[d for d in order if d['g']=='a']; B=[d for d in order if d['g']=='b']
    sc=ax.scatter([d[xk] for d in A],[d[yk] for d in A],c=[d['N'] for d in A],cmap=cmap,norm=norm,zorder=2,**MKa)
    ax.scatter([d[xk] for d in B],[d[yk] for d in B],edgecolors=cmap(norm([d['N'] for d in B])),zorder=2,**MKb)
    if True:  # linear fits + Pearson r
        for grp,lab,ls in [(A,'α-helical','-'),(B,'β-barrel','--')]:
            x=np.array([d[xk] for d in grp]); y=np.array([d[yk] for d in grp]); p=np.polyfit(x,y,1); r=np.corrcoef(x,y)[0,1]
            xx=np.linspace(x.min(),x.max(),50); ax.plot(xx,np.polyval(p,xx),color='0.1',ls=ls,lw=1.2,zorder=4)
            stats[(k,lab)]=(r,p)
            ax.text(0.97,0.04+0.10*(lab=='α-helical'),f"{lab}: r = {r:.2f}",transform=ax.transAxes,fontsize=8.5,ha='right',va='bottom',color='0.1',bbox=dict(fc='white',ec='none',pad=0.5,alpha=0.85),zorder=6)
            ax.plot([0.005,0.025],[0.985-0.10*(lab!='α-helical')-0.03]*2,transform=ax.transAxes,color='0.1',ls=ls,lw=1.0,clip_on=False) if False else None
    ax.set_xlabel(xl,labelpad=2); ax.set_ylabel(yl,rotation=0,ha='right',va='center',labelpad=8)
    ax.set_ylim(10,65 if 'xy' in yk else 62)
    ax.text(-0.30,1.03,'abcd'[k],transform=ax.transAxes,fontsize=16,fontweight='bold',va='bottom')
H=[Line2D([],[],ls='',marker='o',ms=5,mfc='#3C5488',mec='white',label=f'α-helical (n = {na})'),
   Line2D([],[],ls='',marker='^',ms=5.5,mfc='none',mec='#3C5488',mew=0.9,label=f'β-barrel (n = {nb})'),
   Line2D([],[],color='0.1',lw=1.2,label='α fit'),Line2D([],[],color='0.1',lw=1.2,ls='--',label='β fit'),
   Line2D([],[],ls='',label=f's = {SS}')]
fig.legend(handles=H,loc='upper center',ncol=5,fontsize=9,frameon=False,bbox_to_anchor=(0.48,1.0),handletextpad=0.3,columnspacing=1.0)
fig.tight_layout(rect=(0,0,0.9,0.945),h_pad=1.0,w_pad=2.0); fig.align_ylabels(axs[[0,2]]); fig.align_ylabels(axs[[1,3]])
cax=fig.add_axes([0.915,0.30,0.018,0.42]); cb=fig.colorbar(sc,cax=cax); cb.set_label(r'$N_{\mathrm{C\alpha}}$',fontsize=11); cb.ax.tick_params(labelsize=8); cb.outline.set_linewidth(0.6)
for e in ['pdf','svg','png']: fig.savefig(OUT/f"Fig_xi_scatter_L_Rg_s{SS}.{e}",bbox_inches='tight',pad_inches=0.02)
for k,(r,p) in stats.items(): print(k,round(r,3),np.round(p,3))
