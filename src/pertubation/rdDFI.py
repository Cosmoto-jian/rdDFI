#!/usr/bin/env python3
"""
Data processing only (no plotting) based on visualize_fusion_pnas.py.
Computes metrics and saves summary CSV.
Handles both list-of-tuples and flat-list TM region definitions.
- Updated: Input paths and OUT_DIR set to current script directory (./).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import skew

BOTTOM_PCT = 0.20

# ---- TM regions ----
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
    "aa3r":  [(14, 34), (46, 66), (81, 101), (121, 141), (174, 194), (230, 250), (262, 282)],
    "drd2":  [(37, 60), (72, 96), (110, 131), (153, 176), (193, 216), (373, 394), (407, 428)],
    "cxcr3": [(43, 63), (76, 96), (111, 131), (151, 171), (204, 224), (244, 264), (284, 304)],
    "5ht1b": [(46, 66), (79, 99), (114, 134), (154, 174), (197, 217), (317, 337), (353, 373)],
    "5ht2c": [(66, 86), (99, 119), (134, 154), (174, 194), (217, 237), (311, 331), (348, 368)],
    "hrh1":  [(21, 41), (53, 73), (88, 108), (152, 172), (194, 214), (414, 434), (453, 473)],
    "acm2":  [(21, 41), (54, 74), (91, 111), (135, 155), (178, 198), (385, 405), (421, 441)],
    "grm1":  [(593, 613), (626, 646), (656, 676), (699, 719), (733, 753), (779, 799), (813, 833)],
    "acm3":  [(71, 91), (104, 124), (141, 161), (185, 205), (228, 248), (498, 518), (534, 554)],
}

def ensure_pairs(regions):
    if not regions:
        return []
    if isinstance(regions[0], (list, tuple)) and len(regions[0]) == 2:
        return regions
    return [(regions[i], regions[i+1]) for i in range(0, len(regions), 2)]

for key in GPCR_TM_REGIONS:
    GPCR_TM_REGIONS[key] = ensure_pairs(GPCR_TM_REGIONS[key])

# ---- Paths (OUT_DIR set to current script dir ./) ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DFI_DIR = os.path.join(SCRIPT_DIR, "results", "DFI")
MUT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "mutation")
OUT_DIR = SCRIPT_DIR  # 改为当前目录 ./

SUMMARY_CSV = os.path.join(OUT_DIR, "Prediction_models_hit_summary.csv")

METRIC_STYLE = {
    "DFI":             {"color": "#3B82F6", "ls": "-", "label": "DFI"},
    "DFI_membrane":    {"color": "#1E3A8A", "ls": "-", "label": "imANM"},
    "H":               {"color": "#ADFF2F", "ls": "-", "label": "H"},
    "Expert_Deferral": {"color": "#FF69B4", "ls": "-", "label": "RRF"},
}
RUG_METRICS = ["DFI", "DFI_membrane", "H", "Expert_Deferral"]

def _get_uniqueness(d, feat, all_features=['DFI', 'DFI_membrane', 'H']):
    n_sel = max(1, int(round(BOTTOM_PCT * len(d))))
    bottom_idx = set(d.sort_values(feat, ascending=True).iloc[:n_sel].index)
    others = [g for g in all_features if g != feat]
    overlaps = []
    for g in others:
        other_idx = set(d.sort_values(g, ascending=True).iloc[:n_sel].index)
        overlaps.append(len(bottom_idx & other_idx) / n_sel)
    return 1.0 - np.mean(overlaps)

def compute_expert_deferral(df_tm, k=80, uniq_thresh=1.01, skew_guard=0.0, ultra_thresh=1.35):
    d = df_tm.copy()
    n_sel = max(1, int(round(BOTTOM_PCT * len(d))))
    h_uniq = _get_uniqueness(d, 'H')
    max_other_uniq = max(_get_uniqueness(d, 'DFI'), _get_uniqueness(d, 'DFI_membrane'))
    uniq_ratio = h_uniq / max_other_uniq if max_other_uniq > 0 else 999
    h_skew = skew(d['H'].values)

    path_a = uniq_ratio > uniq_thresh and h_skew > skew_guard
    path_b = uniq_ratio > ultra_thresh

    if path_a or path_b:
        selected = d.sort_values('H', ascending=True).iloc[:n_sel]
        return selected['ResI'].astype(int).tolist(), 'H_deferral'
    else:
        features = ['DFI', 'DFI_membrane', 'H']
        for f in features:
            d[f'{f}_rank'] = d[f].rank(method='average', ascending=True)
        d['rrf_score'] = sum(1.0 / (k + d[f'{f}_rank'].values) for f in features)
        selected = d.nlargest(n_sel, 'rrf_score')
        return selected['ResI'].astype(int).tolist(), 'RRF'

def load_protein_data(protein_key):
    dfi_csv = os.path.join(DFI_DIR, f"{protein_key}.csv")
    mut_csv = os.path.join(MUT_RESULTS_DIR, protein_key, f"{protein_key}_mutation_summary.csv")

    if not os.path.exists(dfi_csv) or not os.path.exists(mut_csv):
        return None

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

def evaluate_selection(dfi_tm, metric, mutated_positions_tm, precomputed_sel=None, pct=BOTTOM_PCT):
    n_total = len(dfi_tm)
    n_select = max(1, int(round(pct * n_total)))

    if precomputed_sel is not None:
        selected_positions = precomputed_sel
    else:
        ranked = dfi_tm.sort_values(metric, ascending=True)
        selected_positions = ranked.iloc[:n_select]["ResI"].astype(int).tolist()

    hits = sorted(set(selected_positions) & mutated_positions_tm)
    misses = sorted(set(selected_positions) - mutated_positions_tm)

    precision = len(hits) / n_select if n_select else 0.0
    recall = len(hits) / len(mutated_positions_tm) if mutated_positions_tm else 0.0

    return {
        "n_total": n_total,
        "n_select": n_select,
        "n_mutated": len(mutated_positions_tm),
        "hits": hits,
        "misses": misses,
        "n_hits": len(hits),
        "precision": precision,
        "recall": recall,
    }

def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(GPCR_TM_REGIONS.keys())
    targets = [p for p in targets if p in GPCR_TM_REGIONS]

    summary_rows = []

    for protein_key in targets:
        loaded = load_protein_data(protein_key)
        if loaded is None:
            continue

        dfi_full, mutated_positions_full, pos_mut_count_full = loaded

        tm_residues = set()
        for (a, b) in GPCR_TM_REGIONS[protein_key]:
            tm_residues.update(range(a, b + 1))

        dfi_tm = dfi_full[dfi_full["ResI"].isin(tm_residues)].copy()
        mutated_positions_tm = mutated_positions_full & tm_residues

        if not mutated_positions_tm:
            continue

        ed_sel, _ = compute_expert_deferral(dfi_tm)

        for metric in RUG_METRICS:
            if metric == "Expert_Deferral":
                r = evaluate_selection(dfi_tm, metric, mutated_positions_tm, precomputed_sel=ed_sel)
            else:
                r = evaluate_selection(dfi_tm, metric, mutated_positions_tm)

            summary_rows.append({
                "Protein": protein_key,
                "Metric": metric,
                "N_TM_Residues": r["n_total"],
                "N_Selected": r["n_select"],
                "N_TM_Mutated_Positions": r["n_mutated"],
                "N_Hits": r["n_hits"],
                "Precision": r["precision"],
                "Recall": r["recall"],
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(SUMMARY_CSV, index=False)
        print(f"Summary saved to {SUMMARY_CSV}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()
