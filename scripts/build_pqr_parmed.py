#!/usr/bin/env python3
"""
build_pqr_parmed.py
===================
Build PQR(s) for the capsid protein (and, as a cross-check, sucrose) straight
from the GROMACS text topology using ParmEd -- NO .tpr, NO MDAnalysis.

Per atom, ParmEd gives us:
  * charge   -> atom.charge   (elementary charge, at the simulated protonation)
  * radius   -> atom.rmin     (CHARMM Rmin/2, Angstrom, read from the FF)
  * coords   -> atom.xx/xy/xz (Angstrom, from the .gro passed as xyz=)

Everything is CHARMM-consistent and reflects the actual simulated system.

Run this ON A COMPUTE NODE (loading a ~3M-atom .top takes minutes + several GB).

Usage
-----
  pip install parmed        # pure python, no conda solver
  python build_pqr_parmed.py --top topol.top --gro md_whole.gro \
      --topdir $HOME/gmx-ff --protein-out capsid_protein.pqr \
      --suc-out suc_parmed.pqr

--topdir must be the directory that CONTAINS charmm36.ff (so ParmEd can resolve
the #include lines). If your .top uses absolute include paths, you can omit it.
"""
import argparse
import os
import sys

import numpy as np
import parmed as pmd


def grid_report(atoms, label):
    """Print APBS fglen/cglen suggestions from in-memory coordinates."""
    xyz = np.array([[a.xx, a.xy, a.xz] for a in atoms])
    ext = xyz.max(0) - xyz.min(0)
    fg = ext + 20.0
    cg = 1.6 * fg
    print(f"[{label}] extent A: {ext.round(1)}", file=sys.stderr)
    print(f"[{label}] fglen -> {fg.round(1)}   cglen -> {cg.round(1)}",
          file=sys.stderr)

# CHARMM protein + CHARMM-GUI residue names (incl. HSD/HSE/HSP and the common
# protonation variants). Printed back so you can verify the selection.
PROTEIN_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "HSD", "HSE", "HSP", "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    # protonation / special variants
    "ASPP", "GLUP", "LSN", "CYSD", "CYM", "CYX", "HID", "HIE", "HIP",
}


def write_pqr(atoms, path, resname_override=None):
    net = 0.0
    with open(path, "w") as fh:
        for serial, a in enumerate(atoms, start=1):
            q, r = a.charge, a.rmin           # rmin = CHARMM Rmin/2 (Angstrom)
            net += q
            rn = resname_override or a.residue.name
            fh.write(f"ATOM {serial} {a.name} {rn} {a.residue.number} "
                     f"{a.xx:.3f} {a.xy:.3f} {a.xz:.3f} {q:.4f} {r:.4f}\n")
    return net, serial if atoms else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", required=True)
    ap.add_argument("--gro", required=True)
    ap.add_argument("--topdir", default=None,
                    help="dir containing charmm36.ff (for #include resolution)")
    ap.add_argument("--protein-out", default="capsid_protein.pqr")
    ap.add_argument("--suc-out", default=None,
                    help="optional: also write sucrose PQR (cross-check)")
    ap.add_argument("--suc-resname", default="SUC")
    args = ap.parse_args()

    if args.topdir:
        pmd.gromacs.GROMACS_TOPDIR = args.topdir

    for out in (args.protein_out, args.suc_out):
        if out:
            parent = os.path.dirname(out)
            if parent:
                os.makedirs(parent, exist_ok=True)

    print("loading topology (this is the slow step)...", file=sys.stderr)
    struct = pmd.load_file(args.top, xyz=args.gro)
    print(f"loaded {len(struct.atoms)} atoms, "
          f"{len(struct.residues)} residues", file=sys.stderr)

    # --- RADIUS SANITY CHECK: is atom.rmin really Rmin/2? --------------------
    # Known CHARMM Rmin/2: OG311=1.7650, CG321=2.0100, HGP1=0.2245.
    seen = {}
    for a in struct.atoms:
        t = getattr(a.atom_type, "name", None) or a.type
        if t in ("OG311", "CG321", "HGP1") and t not in seen:
            seen[t] = a.rmin
    if seen:
        print("rmin spot-check (expect OG311~1.765, CG321~2.010, HGP1~0.2245):",
              {k: round(v, 4) for k, v in seen.items()}, file=sys.stderr)
        print("  -> if these are ~2x too big, atom.rmin is Rmin not Rmin/2; "
              "tell me and I'll halve it.", file=sys.stderr)

    # --- protein selection by residue name -----------------------------------
    prot_res = [r for r in struct.residues if r.name in PROTEIN_RESNAMES]
    prot_names = sorted({r.name for r in prot_res})
    prot_atoms = [a for r in prot_res for a in r.atoms]
    print(f"protein: {len(prot_res)} residues, {len(prot_atoms)} atoms; "
          f"resnames={prot_names}", file=sys.stderr)
    if not prot_atoms:
        sys.exit("ERROR: no protein residues matched. Check residue names in "
                 "the topology and extend PROTEIN_RESNAMES.")

    net, n = write_pqr(prot_atoms, args.protein_out)
    print(f"protein PQR: {n} atoms, net charge {net:+.3f} e -> "
          f"{args.protein_out}", file=sys.stderr)

    grid_atoms = list(prot_atoms)
    if args.suc_out:
        suc_atoms = [a for r in struct.residues if r.name == args.suc_resname
                     for a in r.atoms]
        net_s, n_s = write_pqr(suc_atoms, args.suc_out)
        print(f"sucrose PQR: {n_s} atoms, net charge {net_s:+.4f} e "
              f"(expect ~0; should match your itp-derived sucrose.pqr) -> "
              f"{args.suc_out}", file=sys.stderr)
        grid_atoms += suc_atoms

    # APBS grid sizing straight from the coordinates in memory (no file re-read)
    grid_report(grid_atoms, "protein+sucrose" if args.suc_out else "protein")


if __name__ == "__main__":
    main()
