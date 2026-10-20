#!/usr/bin/env python3
"""Select three 4-protein sets from OPM all species (isolated chain imANM, K=1000):
  ab: one per class (bitopic, receptor/channel, transporter, beta-barrel)
  cd: four alpha-helical channels (receptor/channel class, GPCRs excluded)
  ef: four beta-barrels
Constraints for every set:
  (1) Rg increases with length (Cα N), near-proportional: Pearson(N, Rg) >= 0.95, Rg strictly increasing in N order,
      length span N_max/N_min >= 1.5
  (2) Fig2 collapse: φ_xy(ρ_xy) and φ_z(ρ_z) at s = 16 nearly coincide (RMS distance between curves on a common ρ grid)
  (3) ρ*_xy strictly increasing and ρ*_z strictly decreasing with s = 1..32
Score = max pairwise curve distance in the set (+ small weight on ρ*(16) spread)."""
import json, itertools, numpy as np, pandas as pd
from pathlib import Path
DATA = Path(__file__).resolve().parents[1] / "data"
IMANM = DATA / "imANM_results"
S = ['1', '2', '4', '8', '16', '32']
GX = np.linspace(0.15, 2.5, 48); GZ = np.linspace(0.15, 3.5, 68)
t = pd.read_csv(DATA / "representatives_250_850.csv"); t["tag"] = t.pdb.str.lower() + "_" + t.chain
rows, CUR = [], {}
for _, r in t.iterrows():
    f = IMANM / f"{r.tag}.json"
    if not f.exists(): continue
    J = json.load(open(f))
    if not all(s in J["s"] for s in S): continue
    xy = np.array([J["s"][s]["rho_xy"] for s in S]); z = np.array([J["s"][s]["rho_z"] for s in S])
    if np.isnan(xy).any() or np.isnan(z).any(): continue
    if not ((np.diff(xy) > 0).all() and (np.diff(z) < 0).all()): continue
    g = J["s"]["16"]
    rx = np.array(g["xy"]["r"]) / J["L_xy"]; rz = np.array(g["z"]["r"]) / J["L_z"]
    if rx.max() < GX[-1] or rz.max() < GZ[-1]: continue
    CUR[r.tag] = (np.interp(GX, rx, g["xy"]["phi"]), np.interp(GZ, rz, g["z"]["phi"]))
    fam = str(r.family)
    rows.append(dict(tag=r.tag, cls=r.cls, name=r["name"], family=fam, org=r.org, N=int(J["N"]), Rg=r.Rg,
                     gpcr=any(k in fam for k in ["GPCR", "G-protein", "Frizzled", "Smoothened"]),
                     xy1=xy[0], xy16=xy[4], xy32=xy[5], z1=z[0], z16=z[4], z32=z[5], file=f))
DATA.mkdir(parents=True, exist_ok=True)
d = pd.DataFrame(rows).set_index("tag"); d.to_csv(DATA / "pool_mono.csv")
print(d.groupby("cls").size().to_dict(), "channels(non-GPCR):", int(((d.cls == "B_receptor_channel") & ~d.gpcr).sum()))
tags = list(d.index); idx = {k: i for i, k in enumerate(tags)}
X = np.array([CUR[k][0] for k in tags]); Z = np.array([CUR[k][1] for k in tags])
DX = np.sqrt(((X[:, None] - X[None]) ** 2).mean(-1)); DZ = np.sqrt(((Z[:, None] - Z[None]) ** 2).mean(-1))
D = (DX + DZ) / 2

def ok_len(sub):
    a = sub.sort_values("N")
    if a.N.max() / a.N.min() < 1.5: return False
    if (a.N.values[1:] / a.N.values[:-1]).min() < 1.10: return False  # evenly spread lengths
    if not (np.diff(a.Rg.values) > 0.3).all(): return False
    return np.corrcoef(a.N, a.Rg)[0, 1] >= 0.95

def score(c):
    ii = [idx[k] for k in c]; sub = D[np.ix_(ii, ii)]; a = d.loc[list(c)]
    sp = max(np.ptp(a.xy16) / a.xy16.mean(), np.ptp(a.z16) / a.z16.mean())
    return sub.max() + 0.15 * sp, sub.max(), sp

def search(pools, nn=25, top=8):
    """pools: list of 4 candidate tag lists (same list 4x for same-class sets)."""
    same = all(p is pools[0] for p in pools); seen, res = set(), []
    seeds = pools[0]
    for seed in seeds:
        near = [[k for k in sorted(p, key=lambda k: D[idx[seed], idx[k]])[:nn]] for p in pools]
        combos = itertools.combinations(near[0], 4) if same else itertools.product([seed], *near[1:])
        for c in combos:
            if len(set(c)) < 4: continue
            key = frozenset(c)
            if key in seen: continue
            seen.add(key)
            if not ok_len(d.loc[list(c)]): continue
            res.append((score(c), c))
    res.sort(key=lambda x: x[0][0]); out = []
    for (s, dm, sp), c in res[:top]:
        a = d.loc[list(c)].sort_values("N")
        print(f"  D {dm:.3f} sp {sp*100:.1f}% r {np.corrcoef(a.N, a.Rg)[0,1]:.3f} | " +
              " | ".join(f"{k} {x['name'][:18]} N{x.N} Rg{x.Rg:.1f} xy{x.xy16:.2f} z{x.z16:.2f}" for k, x in a.iterrows()))
        out.append(list(a.index))
    return out

cl = lambda c: list(d.index[d.cls == c])
R = {}
CH = list(d.index[(d.cls == "B_receptor_channel") & ~d.gpcr & d.family.str.contains("channel|Channel|uniporter|Anoctamin|Tweety|Polycystin")])
GRP = os.environ.get("SELGRP", "ab,cd,ef").split(",")
if "ab" in GRP: print("== ab: four classes"); R["ab"] = search([cl("A_bitopic"), CH, cl("C_transporter"), cl("D_beta_barrel")], nn=14)
ch = list(d.index[(d.cls == "B_receptor_channel") & ~d.gpcr])
if "cd" in GRP: print("== cd: alpha channels"); R["cd"] = search([ch] * 1 * 4 if False else [ch, ch, ch, ch])
bb = cl("D_beta_barrel")
if "ef" in GRP: print("== ef: beta-barrels"); R["ef"] = search([bb, bb, bb, bb], nn=20)
json.dump(R, open(DATA / f"candidates_{'_'.join(GRP)}.json", "w"), indent=1)
