#!/usr/bin/env python3
"""
surface_radial_density.py
=========================

Number density of WATER and SUCROSE as a function of distance from the capsid
surface (distance = each molecule's minimum distance to the nearest protein
heavy atom). Shows the excluded-volume depletion of sucrose near the surface
and the hydration structuring of water.

The hard part is the shell VOLUME at each distance (the region between r and
r+dr from an irregular surface is not a simple spherical shell). It is estimated
by Monte-Carlo: random points are dropped into the (triclinic) box, their
distance-to-surface is histogrammed the same way, and the fraction in each bin
times the box volume gives that shell's volume. density = count / shell-volume.

Needs the WATER-RETAINED trajectory (not the stripped md_prot_suc.xtc). Heavy —
stride frames; densities are ensemble averages and converge on a subset.

Usage:
  python3 surface_radial_density.py --top md.tpr --xtc md_whole.xtc \\
      --water-atom OH2 --excipient SUC --max-shell 15 --stride 20 \\
      --out surface_radial_density.csv
  # CHARMM water O is usually OH2; GROMACS SPC/TIP3 is OW -- set --water-atom.
"""
import argparse
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from MDAnalysis.lib.mdamath import box_volume, triclinic_vectors


def min_dist(points, ref, box, cutoff):
    """Per-point minimum distance to `ref` within cutoff (inf beyond)."""
    pairs, d = capped_distance(points, ref, max_cutoff=cutoff, box=box, return_distances=True)
    out = np.full(len(points), np.inf)
    if len(pairs):
        order = np.lexsort((d, pairs[:, 0]))
        p, dd = pairs[order], d[order]
        first = np.ones(len(p), bool); first[1:] = p[1:, 0] != p[:-1, 0]
        out[p[first, 0]] = dd[first]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", required=True)
    ap.add_argument("--xtc", required=True)
    ap.add_argument("--protein-sel", default="protein")
    ap.add_argument("--water-resname", default="SOL")
    ap.add_argument("--water-atom", default="OH2", help="water oxygen name (CHARMM OH2 / GROMACS OW)")
    ap.add_argument("--excipient", default="SUC")
    ap.add_argument("--max-shell", type=float, default=15.0)
    ap.add_argument("--bin-width", type=float, default=0.5)
    ap.add_argument("--mc-points", type=int, default=150000, help="random points/frame for shell-volume estimate")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--out", default="surface_radial_density.csv")
    a = ap.parse_args()

    u = mda.Universe(a.top, a.xtc)
    prot = u.select_atoms(f"({a.protein_sel}) and not name H*")
    watO = u.select_atoms(f"resname {a.water_resname} and name {a.water_atom}")
    if len(watO) == 0:
        raise SystemExit(f"no water oxygens (resname {a.water_resname} name {a.water_atom}) "
                         f"-- set --water-atom (CHARMM: OH2, GROMACS: OW).")
    suc = u.select_atoms(f"resname {a.excipient}")
    print(f"protein heavy {len(prot)} | water O {len(watO)} | sucrose {len(suc.residues)} molecules")

    edges = np.arange(0, a.max_shell + a.bin_width, a.bin_width)
    nb = len(edges) - 1
    wat_c = np.zeros(nb); suc_c = np.zeros(nb); vol = np.zeros(nb)
    rng = np.random.default_rng(0); nfr = 0

    for ts in u.trajectory[::a.stride]:
        box = u.dimensions; prot_pos = prot.positions
        wat_c += np.histogram(min_dist(watO.positions, prot_pos, box, a.max_shell), bins=edges)[0]
        scom = suc.center_of_mass(compound="residues")
        suc_c += np.histogram(min_dist(scom, prot_pos, box, a.max_shell), bins=edges)[0]
        # shell volume via Monte-Carlo in the (triclinic) cell
        pts = rng.random((a.mc_points, 3)) @ triclinic_vectors(box)
        vfrac = np.histogram(min_dist(pts, prot_pos, box, a.max_shell), bins=edges)[0] / a.mc_points
        vol += vfrac * box_volume(box)
        nfr += 1
        if nfr % 5 == 0:
            print(f"  frame {ts.frame} ({nfr} processed)")

    r = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        wat_dens = (wat_c / nfr) / (vol / nfr)      # molecules / A^3
        suc_dens = (suc_c / nfr) / (vol / nfr)
    to_M = 1e27 / 6.02214076e23                     # /A^3 -> mol/L
    import csv
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["r_A", "water_M", "sucrose_M", "water_per_A3", "sucrose_per_A3", "n_water", "n_sucrose"])
        for i in range(nb):
            w.writerow([round(r[i], 3), round(wat_dens[i]*to_M, 4), round(suc_dens[i]*to_M, 5),
                        wat_dens[i], suc_dens[i], wat_c[i]/nfr, suc_c[i]/nfr])
    print(f"done ({nfr} frames) -> {a.out}")


if __name__ == "__main__":
    main()
