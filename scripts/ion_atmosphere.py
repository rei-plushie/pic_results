#!/usr/bin/env python3
"""
ion_atmosphere.py
=================

The mobile counter-charge (Na+/Cl-) around the AAV capsid: how the ionic
atmosphere screens the capsid's surface charge, resolved radially and by
icosahedral region (3F/5F). Companion to the formal-charge distribution
(protonation-derived) -- that says where the fixed charge is; this says how the
mobile ions arrange to screen it.

PRIMARY OBSERVABLE -- cumulative counter-ion charge Q(r)
--------------------------------------------------------
For each frame, every ion is assigned its distance to the nearest protein heavy
atom (PBC-correct, minimum image). Q(r) = e * <sum over ions within r of
(z_ion)>  (z = +1 Na, -1 Cl), averaged over frames. Q(r) starts near 0 at the
surface and, as r grows, approaches the NEGATIVE of the capsid's exposed surface
charge once the cloud has fully screened it. The distance where Q(r) plateaus is
the observed screening length -- compare it to the Debye length from the bulk
salt concentration (printed in summary.txt). This needs only ion positions +
distances: no shell-volume normalization, no water. It runs on the
water-stripped md_prot_suc.{gro,xtc} (the `non-Water` group kept NA/CL).

SECONDARY -- per-region screening and the ion<->local-charge correlation
------------------------------------------------------------------------
Near-surface ions (< --near-cutoff) are assigned to their nearest protein
residue's region (3F/5F) and to that residue's formal charge sign. Tests the
screening picture directly: Cl- should pile up on the positive 3F region, Na+
on the negative 5F region. Needs --region-csv (this run's
pic_symmetry_regions/per_residue.csv); omit it for radial-only (e.g. an AAV2/8
salt run without a region map).

OUTPUTS (--output-dir)
----------------------
  ion_shell_profile.csv        r_center_A, na_per_frame, cl_per_frame,
                               net_charge_per_frame, cum_counter_charge_Q_e
  ion_region_summary.csv       region_type: na/cl per frame within near-cutoff,
                               net_ion_charge, n_region_residues, per_100res
  ion_local_charge_crosstab.csv nearest-residue charge sign (+/0/-) x ion type
  summary.txt                  capsid formal charge, bulk ionic strength,
                               Debye length, Q at max shell, per-region net ion

Usage
-----
  python3 ion_atmosphere.py --top md_prot_suc.gro --xtc md_prot_suc.xtc \\
      --region-csv pic_symmetry_regions/per_residue.csv --output-dir ion_out
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from MDAnalysis.lib.mdamath import box_volume

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aav.scripts.pic_calculation_exterior import resid_reset_chain_key  # noqa: E402

ION_Z = {"NA": +1, "CL": -1}
# formal charge from CHARMM protonation-state resname (neutral variants -> 0)
RES_Q = {"ARG": +1, "LYS": +1, "HISH": +1, "ASP": -1, "GLU": -1}


def setup_logger():
    logger = logging.getLogger("ion_atmosphere")
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logger.handlers.clear()
    logger.addHandler(h)
    return logger


def min_dist_and_nearest(ion_pos, prot_pos, box, max_cutoff):
    """Per-ion nearest protein heavy atom: returns (ion_local_idx, min_dist,
    nearest_prot_local_idx) for ions with a neighbor within max_cutoff."""
    pairs, dists = capped_distance(ion_pos, prot_pos, max_cutoff=max_cutoff,
                                   box=box, return_distances=True)
    if len(pairs) == 0:
        return (np.empty(0, int), np.empty(0), np.empty(0, int))
    order = np.lexsort((dists, pairs[:, 0]))          # by ion, then distance
    p = pairs[order]; d = dists[order]
    first = np.ones(len(p), bool)
    first[1:] = p[1:, 0] != p[:-1, 0]                 # first = smallest dist per ion
    return p[first, 0], d[first], p[first, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", required=True, help="Topology (.gro is fine; ions must be present)")
    ap.add_argument("--xtc", required=True)
    ap.add_argument("--region-csv", default=None,
                    help="pic_symmetry_regions/per_residue.csv (enables 3F/5F + local-charge outputs)")
    ap.add_argument("--protein-selection", default="protein")
    ap.add_argument("--na-resname", default="NA")
    ap.add_argument("--cl-resname", default="CL")
    ap.add_argument("--max-shell", type=float, default=30.0, help="Radial range, Angstrom (default 30)")
    ap.add_argument("--bin-width", type=float, default=0.5, help="Shell bin width, Angstrom (default 0.5)")
    ap.add_argument("--near-cutoff", type=float, default=8.0,
                    help="Ions within this of the surface count as near-surface for the region split (default 8)")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    logger = setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)

    u = mda.Universe(args.top, args.xtc)
    protein = u.select_atoms(args.protein_selection)
    prot_heavy = protein.select_atoms("not name H*")
    na = u.select_atoms(f"resname {args.na_resname}")
    cl = u.select_atoms(f"resname {args.cl_resname}")
    logger.info("protein heavy %d | Na %d | Cl %d | %d frames",
                len(prot_heavy), len(na), len(cl), len(u.trajectory))
    if len(prot_heavy) == 0 or (len(na) == 0 and len(cl) == 0):
        raise SystemExit("empty protein or ion selection -- check names / topology.")

    # per-heavy-atom (chain, resid) for nearest-residue region/charge lookup
    chain_key = resid_reset_chain_key(protein)
    if chain_key is None:
        raise SystemExit("resid-reset chain signal unusable -- pass the .gro (not a .tpr); "
                         "see sucrose_hbond_regions.py docstring for why.")
    chain_lut = np.full(len(u.atoms), -1, dtype=np.int32)
    chain_lut[protein.atoms.ix] = chain_key
    hv_chain = chain_lut[prot_heavy.ix]
    hv_resid = prot_heavy.resids

    region_of = charge_of = None
    capsid_q = None
    if args.region_csv:
        reg = pd.read_csv(args.region_csv)
        reg["region_type"] = reg["region_type"].fillna("").replace("", "neither")
        reg["q"] = reg["resname"].map(RES_Q).fillna(0).astype(int)
        region_of = {(int(c), int(r)): rt for c, r, rt in
                     zip(reg["chain"], reg["resid"], reg["region_type"])}
        charge_of = {(int(c), int(r)): q for c, r, q in
                     zip(reg["chain"], reg["resid"], reg["q"])}
        capsid_q = int(reg["q"].sum())
        res_per_region = reg.groupby("region_type").size().to_dict()

    nbins = int(np.ceil(args.max_shell / args.bin_width))
    edges = np.linspace(0, args.max_shell, nbins + 1)
    na_shell = np.zeros(nbins); cl_shell = np.zeros(nbins)
    # per-region near-surface tallies, and ion x local-charge-sign crosstab
    reg_na = {}; reg_cl = {}
    cross = {("+", "Na"): 0, ("+", "Cl"): 0, ("0", "Na"): 0, ("0", "Cl"): 0,
             ("-", "Na"): 0, ("-", "Cl"): 0}
    vol_sum = 0.0; n_frames = 0

    ion_sets = [("Na", na, na_shell), ("Cl", cl, cl_shell)]
    for ts in u.trajectory[args.start:args.stop:args.stride]:
        if args.max_frames and n_frames >= args.max_frames:
            break
        box = u.dimensions
        vol_sum += box_volume(box)
        prot_pos = prot_heavy.positions
        for label, ions, shell in ion_sets:
            if len(ions) == 0:
                continue
            iidx, dmin, near = min_dist_and_nearest(ions.positions, prot_pos, box, args.max_shell)
            if len(iidx) == 0:
                continue
            shell += np.histogram(dmin, bins=edges)[0]
            if region_of is not None:
                sel = dmin < args.near_cutoff
                for np_local in near[sel]:
                    key = (int(hv_chain[np_local]), int(hv_resid[np_local]))
                    rt = region_of.get(key, "unmatched")
                    q = charge_of.get(key, 0)
                    d = reg_na if label == "Na" else reg_cl
                    d[rt] = d.get(rt, 0) + 1
                    sign = "+" if q > 0 else ("-" if q < 0 else "0")
                    cross[(sign, label)] += 1
        n_frames += 1
    logger.info("processed %d frames", n_frames)
    nf = max(n_frames, 1)

    # ---- radial profile + cumulative counter-charge Q(r) ----
    centers = 0.5 * (edges[:-1] + edges[1:])
    net = (cl_shell - na_shell)                       # +1 per Cl, -1 per Na, so this is -(net ion charge)
    net_charge = (na_shell - cl_shell) / nf           # net ION charge per shell per frame (e)
    Q = np.cumsum(net_charge)                         # cumulative counter-ion charge within r
    pd.DataFrame({
        "r_center_A": centers,
        "na_per_frame": na_shell / nf,
        "cl_per_frame": cl_shell / nf,
        "net_charge_per_frame": net_charge,
        "cum_counter_charge_Q_e": Q,
    }).to_csv(os.path.join(args.output_dir, "ion_shell_profile.csv"), index=False)

    # ---- bulk ionic strength + Debye length ----
    mean_vol_A3 = vol_sum / nf
    mean_vol_L = mean_vol_A3 * 1e-27                   # A^3 -> L
    NA_AVO = 6.02214076e23
    c_na = len(na) / (NA_AVO * mean_vol_L)             # mol/L (box-average, includes excess counter-ions)
    c_cl = len(cl) / (NA_AVO * mean_vol_L)
    ionic_strength = 0.5 * (c_na + c_cl)               # 1:1 ions, |z|=1
    debye_A = 3.04 / np.sqrt(ionic_strength) if ionic_strength > 0 else float("nan")  # ~0.304 nm / sqrt(I) -> A

    # ---- per-region + crosstab outputs ----
    if region_of is not None:
        rows = []
        for rt in sorted(set(list(reg_na) + list(reg_cl))):
            n_na = reg_na.get(rt, 0) / nf; n_cl = reg_cl.get(rt, 0) / nf
            nres = res_per_region.get(rt, 0)
            rows.append({"region_type": rt, "na_per_frame": n_na, "cl_per_frame": n_cl,
                         "net_ion_charge_per_frame": n_na - n_cl,
                         "n_region_residues": nres,
                         "net_ion_charge_per_100res": (100 * (n_na - n_cl) / nres) if nres else np.nan})
        pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "ion_region_summary.csv"), index=False)
        pd.DataFrame([{"nearest_res_charge": s, "ion": i, "count": cross[(s, i)]}
                      for (s, i) in cross]).to_csv(
            os.path.join(args.output_dir, "ion_local_charge_crosstab.csv"), index=False)

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as fh:
        fh.write(f"topology   : {args.top}\ntrajectory : {args.xtc}\n")
        fh.write(f"frames     : {n_frames}\n")
        fh.write(f"Na / Cl    : {len(na)} / {len(cl)}  (box-average)\n")
        fh.write(f"mean box V : {mean_vol_A3:.3e} A^3\n")
        fh.write(f"c(Na) c(Cl): {c_na*1000:.1f} / {c_cl*1000:.1f} mM ; ionic strength {ionic_strength*1000:.1f} mM\n")
        fh.write(f"Debye len  : {debye_A:.1f} A  (bulk; compare to where Q(r) plateaus)\n")
        fh.write(f"Q at {args.max_shell:.0f} A : {Q[-1]:+.1f} e  (cumulative counter-ion charge)\n")
        if capsid_q is not None:
            fh.write(f"capsid formal charge (protonation): {capsid_q:+d} e "
                     f"-- full screening would drive Q toward {-capsid_q:+d} e\n")
        for line in _region_lines(region_of, reg_na, reg_cl, nf):
            fh.write(line + "\n")
    logger.info("done -> %s", args.output_dir)


def _region_lines(region_of, reg_na, reg_cl, nf):
    if region_of is None:
        return []
    out = []
    for rt in ["3F", "5F", "neither"]:
        n_na = reg_na.get(rt, 0) / nf; n_cl = reg_cl.get(rt, 0) / nf
        if n_na or n_cl:
            out.append(f"[{rt}] near-surface Na {n_na:.2f}/frame, Cl {n_cl:.2f}/frame "
                       f"-> net ion charge {n_na - n_cl:+.2f} e/frame")
    return out


if __name__ == "__main__":
    main()
