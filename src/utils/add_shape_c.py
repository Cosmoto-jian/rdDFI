#!/usr/bin/env python3
"""
Compute shape descriptors for each high-confidence transmembrane protein,
and store per-protein groups inside the HDF5 file.

Coarse-grained to CA atoms only (consistent with downstream imANM analysis).
Membrane coordinate system: z = membrane normal, x-y = membrane plane.
Length units: Angstrom (mdtraj default is nm, converted on output).

HDF5 structure after run:
  /{category}/
      protein_ids          ← list of all protein IDs in this category
      {Protein_ID}/        ← per-protein group
          @L_xy, @L_z, @b, @c, @SASA, @kappa2, @N_residues

Dependencies: mdtraj, numpy, h5py
"""

import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import h5py
import mdtraj as md

# ── Constants ────────────────────────────────────────────────────────────────
NM_TO_A   = 10.0     # nm  → Å
NM2_TO_A2 = 100.0    # nm² → Å²

MASS_WEIGHTED = False  # CA-only → all equal mass, no weighting needed

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
PDB_ZIP = PROJECT_DIR / "data" / "raw" / "UP000005640_9606.zip"
H5_PATH = PROJECT_DIR / "data" / "process" / "tm_plddt70.h5"


def compute_shape_descriptors(traj_full: md.Trajectory):
    """Coarse-grain to CA and compute membrane-aligned shape descriptors.

    Parameters
    ----------
    traj_full : md.Trajectory
        Full single-frame trajectory from PDB.

    Returns
    -------
    dict : {L_xy, L_z, b, c, SASA, kappa2, N_residues}
        L_xy, L_z in Å; b, c, SASA in Å²; kappa2 dimensionless.
    """
    # ── Coarse-grain: CA only ─────────────────────────────────────────
    sel = traj_full.topology.select("name CA")
    if len(sel) < 3:
        raise ValueError(f"only {len(sel)} CA atoms, need ≥ 3")
    traj = traj_full.atom_slice(sel)

    xyz = traj.xyz                                    # (1, n_atoms, 3) nm
    n_atoms = traj.n_atoms

    # ── weights (mass-weighted or equal) ──────────────────────────────────
    if MASS_WEIGHTED:
        masses = np.array([a.element.mass for a in traj.topology.atoms],
                          dtype=float)
    else:
        masses = np.ones(n_atoms, dtype=float)
    w = masses / masses.sum()

    r = xyz[0]                                        # (n_atoms, 3)
    com = np.sum(w[:, None] * r, axis=0)               # weighted center-of-mass
    d = r - com                                        # centered coords

    # ── gyration tensor G (3×3): G_ab = Σ w_i * d_ia * d_ib ──────────────
    G = np.einsum('i,ia,ib->ab', w, d, d)
    Gxx, Gyy, Gzz = G[0, 0], G[1, 1], G[2, 2]
    Gxy = G[0, 1]

    # ── in-plane 2×2 diagonalization → p1 ≥ p2 ───────────────────────────
    Mplane = np.array([[Gxx, Gxy], [Gxy, Gyy]])
    ev = np.linalg.eigvalsh(Mplane)                    # ascending
    p2, p1 = ev[0], ev[1]
    q = Gzz                                             # normal principal moment

    Rg2 = Gxx + Gyy + Gzz

    # ── shape descriptors (nm / nm²) ─────────────────────────────────────
    L_xy_nm = np.sqrt(Gxx + Gyy)
    L_z_nm  = np.sqrt(Gzz)

    b = q - 0.5 * (p1 + p2)                            # asphericity
    c = p1 - p2                                         # acylindricity (≥ 0)
    kappa2 = (b**2 + 0.75 * c**2) / Rg2**2             # anisotropy

    # ── SASA (Shrake-Rupley) ─────────────────────────────────────────────
    sasa = md.shrake_rupley(traj, mode='atom')          # (1, n_atoms) nm²
    SASA_total = float(sasa.sum())

    return {
        "L_xy":   round(L_xy_nm * NM_TO_A, 4),
        "L_z":    round(L_z_nm  * NM_TO_A, 4),
        "b":      round(b * NM2_TO_A2, 4),
        "c":      round(c * NM2_TO_A2, 4),
        "SASA":   round(SASA_total * NM2_TO_A2, 4),
        "kappa2": round(kappa2, 6),
        "N_residues": traj.n_residues,
    }


def load_pdb_from_zip(zf, member):
    """从 zip 读出 PDB.gz 到临时文件并用 mdtraj 加载。"""
    raw = zf.read(member)
    with tempfile.NamedTemporaryFile(suffix=".pdb.gz", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        return md.load(tmp.name)


def main():
    # ── 0. Delete old geometric attributes ───────────────────────────────
    print(f"Cleaning old attrs from: {H5_PATH}")
    cleaned = 0
    with h5py.File(H5_PATH, "a") as h5f:
        for grp_name in h5f.keys():
            grp = h5f[grp_name]
            for name in grp.keys():
                if name == "protein_ids":
                    continue
                prot_grp = grp[name]
                for attr in ("L_xy", "L_z", "f"):
                    if attr in prot_grp.attrs:
                        del prot_grp.attrs[attr]
                        cleaned += 1
    print(f"  Deleted {cleaned} old attribute entries.")

    # ── 1. Read protein IDs from category groups ─────────────────────────
    print(f"\nLoading protein IDs from: {H5_PATH}")
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

    # ── 2. Compute and write per-protein shape descriptors ───────────────
    total     = sum(len(ids) for ids in category_ids.values())
    processed = 0
    saved     = 0
    missing   = 0
    failed    = 0

    print(f"\nComputing shape descriptors for {total} proteins...")

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

                # ── Load PDB & compute shape descriptors ─────────────
                try:
                    traj_full = load_pdb_from_zip(zf, member)
                    descriptors = compute_shape_descriptors(traj_full)
                except Exception as e:
                    print(f"  ✗ error {pid}: {e}", file=sys.stderr)
                    failed += 1
                    if processed % 1000 == 0:
                        print(f"  [{processed}/{total}] saved {saved}...")
                    continue

                # ── Write per-protein group attrs ─────────────────────
                prot_grp = cat_grp.require_group(pid)
                for key, val in descriptors.items():
                    prot_grp.attrs[key] = val
                saved += 1

                if processed % 200 == 0:
                    print(f"  [{processed}/{total}] saved {saved}...")

        # ── Update root attrs ─────────────────────────────────────────
        h5f.attrs["shape_total"]   = processed
        h5f.attrs["shape_saved"]   = saved
        h5f.attrs["shape_missing"] = missing
        h5f.attrs["shape_failed"]  = failed

    zf.close()

    print(f"\nDone. {saved} saved, {missing} missing, {failed} failed.")


if __name__ == "__main__":
    main()
