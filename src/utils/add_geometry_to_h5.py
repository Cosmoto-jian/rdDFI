#!/usr/bin/env python3
"""
Compute geometric scales (L_xy, L_z, f) from CA coordinates for each
high-confidence transmembrane protein, and store per-protein groups
inside the HDF5 file.

HDF5 structure after run:
  /{category}/
      protein_ids          ← list of all protein IDs in this category
      L_xy / L_z / f       ← existing parallel-array datasets (kept)
      {Protein_ID}/        ← per-protein group
          @L_xy, @L_z, @f, @N_residues

Usage:
  # Access a single protein's geometry directly:
  f["GPCR/A0A096LPK9"].attrs["f"]
"""

import gzip
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import h5py

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
PDB_ZIP  = PROJECT_DIR / "data" / "raw" / "UP000005640_9606.zip"
H5_PATH  = PROJECT_DIR / "data" / "process" / "tm_plddt70.h5"


def geometric_scales(coords_A: np.ndarray):
    """Return (L_xy, L_z, f) from CA coordinates.

    L_xy: in-plane spread  (sqrt of mean lateral variance)
    L_z : normal-direction spread
    f   : L_z / L_xy       (f > 1 → elongated along normal;
                             f < 1 → flattened in-plane)
    """
    xc  = coords_A.mean(axis=0)
    cen = coords_A - xc
    G   = (cen.T @ cen) / len(cen)

    L_xy = float(np.sqrt(G[0, 0] + G[1, 1]))
    L_z  = float(np.sqrt(G[2, 2]))
    f    = L_z / L_xy if L_xy > 0 else 0.0

    return L_xy, L_z, f


def extract_ca_coords(text: str):
    """Extract CA-atom 3D coordinates from decompressed PDB text.

    Returns (N, 3) ndarray or None on failure.
    """
    coords = []
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        coords.append([x, y, z])

    if not coords:
        return None
    return np.array(coords, dtype=np.float64)


def main():
    # ── 1. Read protein IDs from category groups ───────────────────────
    print(f"Loading: {H5_PATH}")
    with h5py.File(H5_PATH, "r") as h5f:
        category_ids = {}
        for grp_name in h5f.keys():
            ds = h5f[grp_name]["protein_ids"]
            protein_ids = [
                pid.decode("utf-8") if isinstance(pid, bytes) else pid
                for pid in ds[:]
            ]
            category_ids[grp_name] = protein_ids
            print(f"  {grp_name}: {len(protein_ids)} proteins")

    # ── 2. Compute and write per-protein geometry ──────────────────────
    total     = sum(len(ids) for ids in category_ids.values())
    processed = 0
    saved     = 0
    missing   = 0
    failed    = 0

    print(f"\nComputing geometric scales for {total} proteins...")

    zf = zipfile.ZipFile(PDB_ZIP)
    members = {n.rsplit("/", 1)[-1]: n for n in zf.namelist()}

    with h5py.File(H5_PATH, "a") as h5f:

        for grp_name, prot_ids in category_ids.items():
            cat_grp = h5f[grp_name]

            for pid in prot_ids:
                processed += 1
                member = members.get(f"{pid}_tr.pdb.gz")

                if member is None:
                    missing += 1
                    if processed % 1000 == 0:
                        print(f"  [{processed}/{total}] saved {saved}...")
                    continue

                coords = extract_ca_coords(
                    gzip.decompress(zf.read(member)).decode("utf-8"))
                if coords is None or len(coords) < 3:
                    failed += 1
                    if processed % 1000 == 0:
                        print(f"  [{processed}/{total}] saved {saved}...")
                    continue

                L_xy, L_z, f = geometric_scales(coords)

                # ── Per-protein group with geometry as attrs ─────────
                prot_grp = cat_grp.require_group(pid)
                prot_grp.attrs["L_xy"]       = round(L_xy, 4)
                prot_grp.attrs["L_z"]        = round(L_z, 4)
                prot_grp.attrs["f"]          = round(f, 4)
                prot_grp.attrs["N_residues"] = len(coords)
                saved += 1

                if processed % 1000 == 0:
                    print(f"  [{processed}/{total}] saved {saved}...")

        # ── Update root attrs ────────────────────────────────────────
        h5f.attrs["geometric_total"]   = processed
        h5f.attrs["geometric_saved"]   = saved
        h5f.attrs["geometric_missing"] = missing
        h5f.attrs["geometric_failed"]  = failed

    zf.close()

    print(f"\nDone. {saved} saved, {missing} missing, {failed} failed.")


if __name__ == "__main__":
    main()
