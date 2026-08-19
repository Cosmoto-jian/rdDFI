#!/usr/bin/env python3
"""
Compute the radius of gyration (Rg) for each high-confidence transmembrane
protein, and store it as a per-protein attribute inside the HDF5 file.

Coarse-grained to CA atoms only, same convention as add_shape_c.py
(equal weights, mdtraj loading, first frame).
Rg is rotation-invariant:  Rg² = Σ wᵢ |rᵢ − r_com|² = tr(G).
Note the identity  Rg² = L_xy² + L_z²  holds for the stored L_xy / L_z.
Length units: Angstrom (mdtraj default is nm, converted on output).

HDF5 structure after run:
  /{category}/{Protein_ID}/
      @Rg
  root attrs: rg_total / rg_saved / rg_missing / rg_failed

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
NM_TO_A = 10.0     # nm  → Å

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[1]
PDB_ZIP = PROJECT_DIR / "data" / "raw" / "UP000005640_9606.zip"
H5_PATH = PROJECT_DIR / "data" / "process" / "tm_plddt70.h5"


def compute_rg(traj_full: md.Trajectory):
    """Coarse-grain to CA and compute the radius of gyration.

    Parameters
    ----------
    traj_full : md.Trajectory
        Full single-frame trajectory from PDB.

    Returns
    -------
    float
        Rg in Å.
    """
    # ── Coarse-grain: CA only ─────────────────────────────────────────
    sel = traj_full.topology.select("name CA")
    if len(sel) < 1:
        raise ValueError("no CA atoms")
    traj = traj_full.atom_slice(sel)

    r = traj.xyz[0]                                    # (n_atoms, 3) nm
    com = r.mean(axis=0)                               # equal weights (CA only)
    d2 = np.sum((r - com) ** 2, axis=1)

    Rg_nm = float(np.sqrt(d2.mean()))                  # = sqrt(tr(G))
    return round(Rg_nm * NM_TO_A, 4)


def load_pdb_from_zip(zf, member):
    """从 zip 读出 PDB.gz 到临时文件并用 mdtraj 加载。"""
    raw = zf.read(member)
    with tempfile.NamedTemporaryFile(suffix=".pdb.gz", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        return md.load(tmp.name)


def main():
    # ── 1. Read protein IDs from category groups ─────────────────────────
    print(f"Loading protein IDs from: {H5_PATH}")
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

    # ── 2. Compute and write Rg for each protein ─────────────────────────
    total     = sum(len(ids) for ids in category_ids.values())
    processed = 0
    saved     = 0
    missing   = 0
    failed    = 0

    print(f"\nComputing Rg for {total} proteins...")

    zf = zipfile.ZipFile(PDB_ZIP)
    members = {
        n.rsplit("/", 1)[-1]: n
        for n in zf.namelist()
        if not n.startswith("__MACOSX")
        and not n.rsplit("/", 1)[-1].startswith("._")
    }

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

                # ── Load PDB & compute Rg ─────────────────────────────
                try:
                    traj_full = load_pdb_from_zip(zf, member)
                    Rg = compute_rg(traj_full)
                except Exception as e:
                    print(f"  ✗ error {pid}: {e}", file=sys.stderr)
                    failed += 1
                    if processed % 1000 == 0:
                        print(f"  [{processed}/{total}] saved {saved}...")
                    continue

                # ── Write per-protein group attr ──────────────────────
                prot_grp = cat_grp.require_group(pid)
                prot_grp.attrs["Rg"] = Rg
                saved += 1

                if processed % 200 == 0:
                    print(f"  [{processed}/{total}] saved {saved}...")

        # ── Update root attrs ─────────────────────────────────────────
        h5f.attrs["rg_total"]   = processed
        h5f.attrs["rg_saved"]   = saved
        h5f.attrs["rg_missing"] = missing
        h5f.attrs["rg_failed"]  = failed

    zf.close()

    print(f"\nDone. {saved} saved, {missing} missing, {failed} failed.")


if __name__ == "__main__":
    main()
