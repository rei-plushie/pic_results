#!/usr/bin/env python3
"""
pic_calculation_exterior.py
============================

Variant of pic_calculation.py that restricts the preferential interaction
coefficient (Gamma_23) calculation to EXTERIOR solvent only -- excipient
(and water) molecules outside the capsid shell, in the bulk formulation
medium. Interior (luminal) solvent is excluded entirely, because it is not
in free exchange with the bulk and is not informative for formulation
stability (a sealed or slow-exchanging interior population tells you
nothing about how the excipient behaves at the surface that actually
matters for aggregation/stability).

Everything else -- the per-residue algorithm, PBC unwrap (residue split
across the box boundary), chain-level rigid realignment (whole monomers in
the wrong periodic image), the dynamic bruteforce/nsgrid method choice
(avoids the OOM failure mode of a naive capped_distance call at capsid
scale), the hybrid-36 PDB writer (works past the 99999-atom/9999-resSeq
PDB limits), the MPI driver and its no-MPI local fallback -- is unchanged
from pic_calculation.py. See that file's docstring for the full history of
why each of those exists. This file only adds the interior/exterior
classification layer described below; search for "EXTERIOR-ONLY" to find
every place that differs from pic_calculation.py.

CLASSIFICATION METHOD
----------------------
AAV (and similar T=1 icosahedral) capsids are close enough to spherically
symmetric that the robust, cheap way to separate "inside the shell" from
"outside the shell" is RADIAL DISTANCE FROM THE CAPSID'S OWN CENTROID --
not an explicit 3D surface/point-in-polygon test. A surface-based test
would need to handle the capsid's actual (porous, faceted, dynamically
fluctuating) geometry correctly, including the 5-fold symmetry channels
that are real, physical openings through the shell -- exactly the kind of
feature that makes naive ray-casting/winding-number inside-tests noisy
right where it matters. Radial distance from the centroid sidesteps this
entirely: it doesn't care about the shell's exact shape, only how far a
point is from the assembly's center, so channel pores or surface roughness
don't create classification artifacts.

The shell's own radial extent (inner/outer radius) is not hardcoded --
different serotypes and constructs differ, and getting it wrong from a
literature number would silently mis-classify everything. Instead it is
CALIBRATED from the trajectory itself, once, before the main loop, using
the water radial NUMBER-DENSITY profile (molecules per unit shell volume)
around the capsid centroid, averaged over --n-shell-calibration-frames
sampled frames (every MPI rank samples the identical deterministic set,
so no broadcast is needed to keep ranks in sync).

An earlier version of this calibration instead took a percentile of
residue radial POSITIONS (e.g. the 98th percentile as the outer boundary).
That was confirmed on a real capsid run to be too conservative: P98
calibrates against only the most protruding ~2% of residues, so with a
practical search radius (10-20 A) roughly 95% of ALL residues could never
geometrically reach far enough to touch the exterior-classified pool,
regardless of their real local chemistry -- the vast majority of reported
near-zero Gamma_23 values were a geometric artifact of the threshold
(confirmed via point-biserial correlation between "how far inside the
P98 mark a residue sits" and "near-zero Gamma_23", r=0.45, p~0), not a
real absence of preference.

The density profile is physically grounded instead: water density
necessarily DIPS wherever protein atoms displace it (the shell itself)
and rises back toward the bulk value outside it (and, if the lumen is
solvated, plateaus at some lumen-characteristic value on the inside).
See calibrate_shell_geometry for the full procedure. The outer boundary
is placed at the density MINIMUM itself -- deliberately "closer inside"
rather than waiting for full bulk-density recovery further out, so most
of the surface can actually reach exterior solvent within a practical
search radius -- and the inner boundary analogously on the lumen side of
the same trough. A diagnostic plot of the profile with both boundaries
marked is saved every run (pic_<EXCIPIENT>_shell_density.png) -- always
check it before trusting the classification, the same way you'd check
find_rstar's auto-detected r* against the saved r-profile plot.

A solvent molecule is then classified, every frame, from its own COM's
radial distance from that frame's capsid centroid:
  - EXTERIOR  if radial distance > (outer radius + --shell-margin)
  - INTERIOR  if radial distance < (inner radius - --shell-margin)
  - AMBIGUOUS otherwise (co-located with the protein shell itself, e.g. in
    a pore or transiently surface-bound) -- excluded from BOTH categories,
    not counted anywhere. This is a deliberate choice: the point of this
    script is a clean, unambiguous "true bulk-exposed" reservoir, and
    solvent sitting right in the shell is not clearly either.

Only EXTERIOR-classified excipient AND water are used in the Gamma_23
calculation: the depletion-corrected bulk totals n1_total/n3_total (Eq. 3
of Shukla et al.) are recomputed EVERY FRAME as that frame's exterior
water/excipient counts (not a fixed topology-wide constant, since exactly
which molecules are "exterior" can shift slightly frame to frame), and the
per-residue local-domain search only ever sees exterior solvent COMs as
candidates. A residue facing the capsid's INTERIOR lumen will then
naturally find zero local exterior solvent and get a degenerate/near-zero
Gamma_23 -- this is the correct, expected outcome (that residue has no
access to the bulk formulation medium), not a bug, but it means a
near-zero Gamma_23 in the output should be read alongside the per-residue
"facing" column (see OUTPUTS) rather than alone.

LIMITATIONS
-----------
This assumes a roughly closed, roughly spherical shell (true for AAV and
other T=1 icosahedral capsids). It is not the right approach for a very
different geometry (e.g. a rod-shaped or filamentous assembly), where
"radial distance from one centroid" doesn't correspond to "distance from a
shell". Also: if the topology doesn't carry usable chain/segid information
to distinguish the assembly's separate monomers, the same caveat that
applies to pic_calculation.py's chain-realignment step applies here too
(the log will say so at startup).

OUTPUTS
-------
  pic_<EXCIPIENT>_exterior.pkl                 -- full per-residue + whole-molecule
                                                   arrays, plus shell calibration info
  pic_<EXCIPIENT>_exterior_per_residue.csv     -- resid, resname, gamma23, sem, r*,
                                                   PLUS mean_radial_distance_angstrom
                                                   and facing (outer/inner) -- use
                                                   `facing` to interpret near-zero
                                                   values correctly (see above)
  pic_<EXCIPIENT>_exterior_vs_r.csv            -- whole-molecule r-profile (validation)
  pic_<EXCIPIENT>_exterior_summary.png         -- r-profile + per-residue bar, colored
                                                   by outer/inner facing
  pic_<EXCIPIENT>_shell_density.png            -- water radial density profile with the
                                                   calibrated inner/outer boundary marked --
                                                   check this before trusting the run
  pic_<EXCIPIENT>_exterior_bfactor.pdb         -- reference-frame structure, B-factor =
                                                   per-residue Gamma_23 (inner-facing
                                                   residues will mostly read ~0)

Usage: identical to pic_calculation.py, with three extra optional flags
(--shell-margin, --density-bin-width, --n-shell-calibration-frames):
    mpirun -np <N> python3 pic_calculation_exterior.py \\
        --gro md.gro --xtc md_noPBC.xtc \\
        --excipient SUC --radius 20.0 \\
        --output-dir pic_results_exterior

    python3 pic_calculation_exterior.py --gro md.gro --xtc md_noPBC.xtc \\
        --excipient SUC --radius 20.0 --max-frames 20 --output-dir /tmp/test
"""
import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np

# matplotlib must use a non-interactive backend before pyplot is imported --
# there is no display on HPC compute nodes.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import MDAnalysis as mda  # noqa: E402
from MDAnalysis.lib.distances import capped_distance, minimize_vectors  # noqa: E402

try:
    from mpi4py import MPI
    _HAS_REAL_MPI = True
except Exception:
    _HAS_REAL_MPI = False

    class _SerialComm:
        """Minimal stand-in for mpi4py's Comm, used when mpi4py or an MPI
        runtime isn't available (e.g. local testing on a laptop). Only the
        subset of the API this script actually calls is implemented."""

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def Barrier(self):
            pass

        def allreduce(self, value, op=None):
            return value

        def reduce(self, value, op=None, root=0):
            return value

        def gather(self, value, root=0):
            return [value]

        def Abort(self, code=1):
            sys.exit(code)

    class MPI:  # noqa: N801 -- mimicking the mpi4py.MPI module surface
        COMM_WORLD = _SerialComm()
        SUM = "SUM"
        MAX = "MAX"
        MIN = "MIN"


# --------------------------------------------------------------------------
# Bondi (1964) van der Waals radii, Angstrom. Used both for the original
# atom-level VdW correction and, folded in via residue_radius, for
# the residue-sphere radius. Only elements expected in a standard CHARMM36
# protein (no cofactors / metals) are included; anything else triggers a
# logged warning and falls back to the carbon radius.
# --------------------------------------------------------------------------
BONDI_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
}
DEFAULT_VDW = 1.70  # fallback (carbon) for unrecognized elements, with a warning

# Formal charges for the two ionic excipients, used only for the optional
# charge-balance diagnostic (see --charge / build_charge_profile).
ION_FORMAL_CHARGE = {"NA": +1.0, "CL": -1.0}

# Rough per-frame cost with the residue-sphere search (measured ~0.2-0.25s on
# an ~280k-atom / 518-residue system; scales with system size, not a hard
# constant). Used only for the rank-count advisory below, not for correctness.
SECONDS_PER_FRAME_ESTIMATE = 0.25
# Below this many frames/rank, fixed per-rank overhead (Universe/topology
# load, MPI startup) starts to dominate over actual compute -- see the
# advisory logged where n_frames and size are both known. This matters when
# requesting a large, HPC-scale --ntasks (e.g. up to 384) against a short
# trajectory or a quick test run.
MIN_USEFUL_FRAMES_PER_RANK = 20


# ==========================================================================
# Core numerical routines (shared with pic_calculation.py; see that file
# for the full rationale behind PBC unwrap, chain realignment, and the
# dynamic bruteforce/nsgrid method choice)
# ==========================================================================

def guess_element(atom_name):
    """Guess element from a standard CHARMM/AMBER protein atom name.

    Strips leading digits (e.g. '1HB1' -> 'HB1') and takes the first
    alphabetic character. This is correct for standard protein atom naming
    (CA, CB, OG1, ND2, SD, HB1, ... all genuinely start with their element's
    letter) and deliberately does NOT use a generic two-letter-element
    guesser, because within a protein selection 'CA' means alpha-carbon
    (element C), not the metal ion calcium -- a generic guesser can get this
    backwards. This function is only ever applied to the `protein` selection.
    """
    name = atom_name.lstrip("0123456789")
    return name[0].upper() if name else "?"


def get_protein_vdw_radii(protein_ag, logger):
    """Per-atom VdW radius array for the protein selection, Bondi scale."""
    elements = [guess_element(n) for n in protein_ag.names]
    unknown = sorted(set(e for e in elements if e not in BONDI_RADII))
    if unknown:
        logger.warning(
            "Protein atom elements not in Bondi table (using C=%.2f A as "
            "fallback): %s -- verify these aren't cofactors/metals that "
            "need their own radius.", DEFAULT_VDW, unknown
        )
    return np.array([BONDI_RADII.get(e, DEFAULT_VDW) for e in elements], dtype=float)


def validate_masses(ag, label, logger):
    """Fail loudly (not silently) if mass guessing produced garbage.

    True center-of-mass requires real per-atom masses. MDAnalysis guesses
    masses from guessed elements when reading a bare .gro (no force-field
    info), and that guess is normally fine for the elements present here
    (H, C, N, O, S, Na, Cl) -- but a silent guessing failure (e.g. all
    masses defaulting to 0 or 1) would corrupt every downstream COM, so we
    check rather than assume.
    """
    m = ag.masses
    if m.size == 0:
        raise ValueError(f"[{label}] selection is empty -- check residue name / topology.")
    if np.any(~np.isfinite(m)) or np.any(m <= 0):
        bad = np.unique(ag.names[~np.isfinite(m) | (m <= 0)])
        raise ValueError(
            f"[{label}] {bad.size} atom name(s) have invalid mass (<=0 or NaN), "
            f"mass guessing likely failed for: {list(bad)[:10]}"
        )
    logger.info("[%s] %d atoms, mass range %.3f-%.3f amu (OK)",
                label, m.size, m.min(), m.max())


def build_group_meta(resindices, n_expected, label, logger):
    """One-time (topology-only, not per-frame) grouping metadata for a
    selection's residues/molecules: a local 0..n_groups-1 index per atom,
    plus a (sort_order, group_starts, first_atom_idx) triple that lets
    every later per-frame reduction (COM, radius, PBC unwrap anchor) run
    as a vectorized sort+reduceat instead of np.*.at scatter ops, which
    don't vectorize well once atom counts reach into the hundreds of
    thousands (a full AAV capsid, not just a monomer).
    """
    uniq_resindex, local_idx = np.unique(resindices, return_inverse=True)
    n_groups = uniq_resindex.size
    if n_groups != n_expected:
        raise ValueError(
            f"[{label}] grouping mismatch: {n_groups} unique resindices in "
            f"atom selection vs {n_expected} expected residues/molecules."
        )
    sort_order = np.argsort(local_idx, kind="stable")
    local_idx_sorted = local_idx[sort_order]
    group_starts = np.searchsorted(local_idx_sorted, np.arange(n_groups))
    first_atom_idx = sort_order[group_starts]  # one anchor atom per group, for PBC unwrap
    logger.info("[%s] %d atoms grouped into %d residues/molecules", label, resindices.size, n_groups)
    return {
        "local_idx": local_idx, "n_groups": n_groups,
        "sort_order": sort_order, "group_starts": group_starts,
        "first_atom_idx": first_atom_idx,
    }


def unwrap_positions(positions, group_meta, box):
    """Minimum-image unwrap of each residue/molecule relative to its own
    first atom (by file order), so a group split across a periodic
    boundary doesn't corrupt its center of mass or extent.

    Uses MDAnalysis's own minimize_vectors (correct for triclinic boxes,
    not just orthorhombic) rather than a manual round() formula.
    """
    if box is None or np.any(np.asarray(box[:3]) <= 0):
        return positions
    anchor = positions[group_meta["first_atom_idx"]][group_meta["local_idx"]]
    delta = minimize_vectors(positions - anchor, box)
    return anchor + delta


def rigid_align_chains(positions, chain_meta, box):
    """Rigidly shift each chain/segment into the same periodic image as a
    single reference chain (the one containing the very first atom).

    GROMACS's typical per-molecule wrapping (`-pbc mol`) wraps each chain
    independently by its own centroid, so for a multi-chain assembly (e.g.
    a 60-subunit capsid) different, non-covalently-linked chains can each
    land in a different periodic image even though every chain is
    individually whole. This computes ONE minimum-image translation per
    chain (from that chain's centroid to the reference chain's centroid)
    and applies it uniformly to every atom of the chain, preserving
    whatever internal structure the per-residue unwrap already
    established.
    """
    if box is None or np.any(np.asarray(box[:3]) <= 0):
        return positions
    n_chains = chain_meta["n_groups"]
    local_idx = chain_meta["local_idx"]
    counts = np.bincount(local_idx, minlength=n_chains).astype(float)
    centroid = np.empty((n_chains, 3), dtype=float)
    for dim in range(3):
        centroid[:, dim] = np.bincount(local_idx, weights=positions[:, dim], minlength=n_chains) / counts
    reference = centroid[0]  # group 0 = chain containing the first atom in file order
    raw_offset = centroid - reference
    shift = minimize_vectors(raw_offset, box) - raw_offset
    return positions + shift[local_idx]


def resid_reset_chain_key(protein_ag):
    """Alternative chain-boundary signal, independent of segid/chainID
    metadata: a new chain starts wherever resid decreases relative to the
    previous atom. True for most GROMACS/CHARMM-GUI-prepared multi-chain
    topologies, which number residues 1..N independently per chain rather
    than globally across the whole assembly.

    Added after a real capsid run: segid/chainID metadata silently merged
    two physically separate monomers into one group there, which
    rigid_align_chains then could not correctly separate (it can only
    apply ONE rigid shift per group -- two genuinely different chains
    sharing a group get averaged/shifted together incorrectly). resid
    resets are a purely topological signal that doesn't depend on that
    metadata being right, and correctly identified all 60 chains on that
    dataset when segid/chainID did not. See the chain-candidate diagnostic
    logged at startup, and prefer this signal when it's usable.

    Returns a per-atom chain index array, or None if the signal looks
    unusable (resid never decreases -- e.g. globally continuous residue
    numbering across the whole selection, in which case this can't find
    any boundaries).
    """
    resids = protein_ag.resids.astype(np.int64)
    is_boundary = np.concatenate([[True], np.diff(resids) < 0])
    if int(is_boundary.sum()) <= 1:
        return None
    return np.cumsum(is_boundary) - 1


def unwrap_protein_positions(raw_positions, res_meta, chain_meta, box):
    """Full PBC correction for the protein selection: per-residue unwrap
    (fixes a residue split across the boundary) followed by rigid
    per-chain realignment (fixes a whole monomer sitting in a different
    periodic image than the rest of the assembly). Order matters --
    residue-level must run first so each chain's rigid shift is computed
    from already-cohesive residue positions.
    """
    res_unwrapped = unwrap_positions(raw_positions, res_meta, box)
    return rigid_align_chains(res_unwrapped, chain_meta, box)


def mass_weighted_com(positions, masses, group_meta):
    """Per-group (residue/molecule) center of mass via bincount -- a fast,
    vectorized grouped weighted sum, used instead of
    AtomGroup.center_of_mass(compound=...) so it can run on already
    PBC-unwrapped positions instead of the raw (possibly split) ones."""
    n_groups = group_meta["n_groups"]
    local_idx = group_meta["local_idx"]
    total_mass = np.bincount(local_idx, weights=masses, minlength=n_groups)
    com = np.empty((n_groups, 3), dtype=float)
    for dim in range(3):
        com[:, dim] = np.bincount(local_idx, weights=masses * positions[:, dim], minlength=n_groups)
    com /= total_mass[:, None]
    return com


def group_max_reduce(values, group_meta):
    """Per-group max via sort + reduceat -- vectorized equivalent of
    np.maximum.at (which doesn't vectorize well at large atom counts)."""
    sorted_vals = values[group_meta["sort_order"]]
    return np.maximum.reduceat(sorted_vals, group_meta["group_starts"])


def residue_radius(protein_pos_unwrapped, res_com, group_meta, protein_vdw, logger=None):
    """Conservative per-residue sphere radius:
        max_{atoms j in r} ( ||pos_j - res_com[r]|| + vdw[j] )
    i.e. the same "atom is a VdW sphere" idea as the original whole-protein
    method, applied to bound a residue instead of a single atom.
    """
    delta = protein_pos_unwrapped - res_com[group_meta["local_idx"]]
    atom_reach = np.linalg.norm(delta, axis=1) + protein_vdw
    res_radius = group_max_reduce(atom_reach, group_meta)
    if logger is not None and res_radius.max() > 15.0:
        logger.warning(
            "Residue sphere radius %.2f A is unusually large (>15 A) -- "
            "check for a chain break or a non-protein atom leaking into "
            "the protein selection (should be rare now that positions are "
            "PBC-unwrapped per residue before this check).", res_radius.max()
        )
    return res_radius


def frame_residue_histograms(solvent_com, res_com, res_radius, n_res, n_bins, bin_width, r_max):
    """Per-residue distance histogram for one solvent type in one frame.

    Returns (res_hist, molecule_min_dist):
      res_hist          -- (n_res, n_bins) int counts, res_hist[r, k] = number
                            of solvent COMs whose surface distance to
                            residue r's sphere falls in bin k.
      molecule_min_dist -- (n_solvent,) float, the minimum surface distance
                            from each solvent molecule to ANY residue sphere
                            within r_max (np.inf if none found). Used only
                            for the whole-molecule validation curve.

    EXTERIOR-ONLY NOTE: `solvent_com` is expected to already be filtered to
    exterior-classified molecules by the caller -- this function itself has
    no notion of interior/exterior, it just searches whatever COMs it's
    given. A residue facing the capsid interior will therefore find no
    candidates here (all its would-be neighbors were filtered out upstream)
    and get an all-zero histogram, which is the correct/expected outcome.
    """
    n_solvent = solvent_com.shape[0]
    res_hist = np.zeros((n_res, n_bins), dtype=np.int64)
    molecule_min_dist = np.full(n_solvent, np.inf, dtype=float)
    if n_solvent == 0 or n_res == 0:
        return res_hist, molecule_min_dist

    cutoff = r_max + res_radius.max()
    # method='bruteforce' is faster (2-5x, benchmarked) at small
    # reference-point counts, but _bruteforce_capped computes the FULL
    # DENSE distance_array(solvent_com, res_com) before applying the
    # cutoff mask -- memory scales with n_solvent * n_res regardless of
    # how many pairs actually survive. MDAnalysis's own auto-selection
    # (_determine_method) refuses to pick it once that product reaches
    # 1e8; we use a 2x-safety-margin threshold below that (5e7 elements
    # -> <=400MB for the dense float64 matrix) so bruteforce only ever
    # runs when it's provably cheap, and fall back to the default
    # heuristic's grid-based method (which scales with actual output size,
    # not n_solvent * n_res) otherwise. Forcing bruteforce unconditionally
    # OOM-killed a real capsid-scale job at ~124 GB for a single rank.
    _BRUTEFORCE_SAFE_LIMIT = 5e7
    n_pairs_bound = n_solvent * n_res
    method = "bruteforce" if n_pairs_bound < _BRUTEFORCE_SAFE_LIMIT else None
    pairs, dists = capped_distance(
        solvent_com, res_com, cutoff, box=None, return_distances=True, method=method
    )
    if pairs.size == 0:
        return res_hist, molecule_min_dist

    sol_idx = pairs[:, 0]
    res_idx = pairs[:, 1]
    adjusted = np.clip(dists - res_radius[res_idx], 0.0, None)

    # cutoff was inflated by the GLOBAL max residue radius so a single
    # capped_distance call could serve every residue; for residues with a
    # smaller radius this admits candidates whose true adjusted distance
    # exceeds r_max, which we now drop.
    in_range = adjusted < r_max
    sol_idx = sol_idx[in_range]
    res_idx = res_idx[in_range]
    adjusted = adjusted[in_range]

    if adjusted.size:
        np.minimum.at(molecule_min_dist, sol_idx, adjusted)

        bin_idx = np.minimum((adjusted / bin_width).astype(np.int64), n_bins - 1)
        combined = res_idx * n_bins + bin_idx
        counts = np.bincount(combined, minlength=n_res * n_bins)
        res_hist = counts.reshape(n_res, n_bins)

    return res_hist, molecule_min_dist


def gamma_from_cumulative_counts(n1_r, n3_r, n1_total, n3_total):
    """Eq. 3 of Shukla et al., vectorized over any leading shape.

    n1_r, n3_r: cumulative local counts (..., n_bins). n1_total, n3_total:
    scalars -- the bulk reservoir size. EXTERIOR-ONLY NOTE: unlike
    pic_calculation.py (where these are fixed topology-wide molecule
    counts), here the caller passes the CURRENT FRAME's exterior-only
    counts, since which molecules count as "exterior" can shift slightly
    frame to frame.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        bulk_ratio = (n3_total - n3_r) / (n1_total - n1_r)
        gamma = n3_r - n1_r * bulk_ratio
    return gamma


def find_rstar(r_values, gamma_mean, tail_frac=0.3, tol_frac=0.05):
    """Heuristic starting estimate for r* (plateau onset).

    Scans r from large to small; r* is the smallest r beyond which
    Gamma_23(r) stays within tol_frac of the plateau estimate (the median
    of the outermost tail_frac of bins) for the rest of the curve.

    This is a CONVENIENCE STARTING POINT ONLY. Always check this against
    the saved plot before trusting the reported Gamma_23, especially for
    inner-facing residues (see the `facing` column) where the curve may be
    flat at ~0 by construction and r* is close to meaningless.
    """
    n = len(gamma_mean)
    tail_n = max(3, int(n * tail_frac))
    finite_tail = gamma_mean[-tail_n:][np.isfinite(gamma_mean[-tail_n:])]
    if finite_tail.size == 0:
        return r_values[-1], len(r_values) - 1
    plateau_est = np.median(finite_tail)
    tol = max(tol_frac * abs(plateau_est), 1e-6)

    for i in range(n):
        window = gamma_mean[i:]
        window = window[np.isfinite(window)]
        if window.size == 0:
            continue
        if np.all(np.abs(window - plateau_est) <= tol):
            return r_values[i], i
    return r_values[-1], n - 1  # no clean plateau found -> flag via max r


def format_eta(seconds):
    """Human-readable HH:MM:SS from a float seconds value (or '--:--:--' if
    not yet estimable, e.g. before any frames have completed)."""
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def split_frame_range(n_frames, rank, size):
    """Contiguous frame block for this rank (first `n_frames % size` ranks
    get one extra frame). Returns (start, stop) as a half-open range;
    (n, n) i.e. an empty range if this rank has more workers than frames."""
    base = n_frames // size
    rem = n_frames % size
    start = rank * base + min(rank, rem)
    stop = start + base + (1 if rank < rem else 0)
    return start, stop


def build_charge_profile(r_values, n_excipient_r_whole, starting_charge, excipient_name):
    """Optional diagnostic: cumulative net charge Q(r) = starting_charge +
    (formal ion charge) * n_excipient_r(t), time-averaged, using the
    whole-molecule (residue-sphere-derived, exterior-only) local counts.

    This is NOT folded into the Gamma_23 result -- it is a separate
    electroneutrality/Donnan bookkeeping diagnostic, only meaningful for the
    ionic excipients (NA, CL), and only as accurate as the supplied
    starting_charge (net protein formal charge).
    """
    if excipient_name not in ION_FORMAL_CHARGE:
        return None
    z = ION_FORMAL_CHARGE[excipient_name]
    return r_values, starting_charge + z * n_excipient_r_whole


# --------------------------------------------------------------------------
# EXTERIOR-ONLY: capsid shell calibration and radial classification.
# --------------------------------------------------------------------------

def calibrate_shell_geometry(u, protein, protein_meta, protein_chain_meta, protein_masses,
                              water, water_meta, water_masses,
                              n_res, n_calib_frames, density_bin_width, logger, rank):
    """Estimate the capsid shell's inner/outer radii from the water radial
    NUMBER-DENSITY profile around the capsid centroid.

    An earlier version of this calibration used a percentile of residue
    radial POSITIONS (e.g. the 98th percentile as the "outer" boundary).
    That was confirmed on a real capsid run to be too conservative: P98
    calibrates against only the most protruding ~2% of residues, so with a
    practical search radius (10-20 A), ~95% of ALL residues geometrically
    could never reach far enough to touch the exterior-classified solvent
    pool, regardless of their real local chemistry -- the vast majority of
    reported "near-zero" Gamma_23 values were a geometric artifact of the
    threshold, not a real absence of preference (confirmed via a point-
    biserial correlation between "distance inside the P98 mark" and "near-
    zero Gamma_23", r=0.45, p~0).

    This version is physically grounded instead: water number density
    rho(r) = (water molecules in shell [r, r+dr)) / (shell volume),
    averaged over --n-shell-calibration-frames, necessarily DIPS wherever
    protein atoms physically displace water (the shell itself) and rises
    back toward the bulk value outside it (and, if the lumen is solvated,
    plateaus at some lumen-characteristic value on the inside). The outer
    boundary is placed at the density MINIMUM itself -- deliberately
    "closer inside" rather than waiting for full bulk-density recovery
    further out, so most of the surface can actually reach exterior
    solvent within a practical search radius. The inner boundary is placed
    analogously on the lumen side of the same trough.

    Every rank runs this independently against the identical deterministic
    frame sample, so no broadcast is needed to keep ranks in sync.

    Returns (r_inner, r_outer, res_radial_mean, density_profile) where
    res_radial_mean is each residue's own mean radial distance across the
    sampled frames (used for the per-residue outer/inner "facing"
    diagnostic) and density_profile is (centers, density) for diagnostics/
    plotting.
    """
    n_frames_all = len(u.trajectory)
    n_calib = max(1, min(n_calib_frames, n_frames_all))
    calib_indices = np.unique(np.linspace(0, n_frames_all - 1, n_calib, dtype=int))

    radii_per_frame = np.empty((len(calib_indices), n_res), dtype=float)
    water_radial_samples = []
    for k, fidx in enumerate(calib_indices):
        u.trajectory[int(fidx)]
        box = u.dimensions  # may fluctuate frame-to-frame under NPT
        # Protein positions are used AS-IS (no unwrap/chain-realign): on a
        # -pbc-whole-processed trajectory the capsid is already one
        # contiguous object (verified: no chain needs a minimum-image shift
        # relative to the overall centroid), it just isn't necessarily
        # wrapped back into [0, box) -- some chains legitimately sit past
        # the nominal box edge while still being correctly connected to the
        # rest. Reintroducing protein-side unwrap/rigid-chain-alignment
        # here previously did more harm than good: it flagged and shifted
        # chains that were already fine (far from an arbitrary reference
        # point, not actually mispositioned) while leaving the real
        # problem untouched -- see the chain-by-chain diagnostic that
        # motivated this fix.
        pos = protein.positions
        res_com = mass_weighted_com(pos, protein_masses, protein_meta)
        centroid = res_com.mean(axis=0)
        radii_per_frame[k] = np.linalg.norm(res_com - centroid, axis=1)

        # Water DOES need re-imaging, though: it's wrapped by GROMACS into
        # the canonical [0, box) range independent of where the protein
        # ended up, so for protein sitting past the box edge, its true
        # nearby water reads as ~a box-length away in raw coordinates.
        # Verified directly: for the worst-affected chain, only 205/517
        # residues had ANY water neighbor within 15 A using raw water
        # coordinates; re-imaging each water molecule to its nearest
        # periodic copy relative to the centroid brought that to 517/517.
        water_pos = water.positions
        water_com = mass_weighted_com(water_pos, water_masses, water_meta)
        water_com = centroid + minimize_vectors(water_com - centroid, box)
        water_radial_samples.append(np.linalg.norm(water_com - centroid, axis=1))

    res_radial_mean = radii_per_frame.mean(axis=0)
    all_water_radial = np.concatenate(water_radial_samples)

    # Bound the density search window using residue positions. Deliberately
    # ASYMMETRIC padding: generous on the outside (the true bulk transition
    # legitimately extends somewhat beyond the outermost residue, by
    # roughly a solvation shell's worth), but tight on the inside (no
    # inward padding past the innermost residues) -- for a genuinely hollow,
    # solvated shell the lumen's own bulk-like plateau is reached well
    # inside the innermost shell residues already, and padding further in
    # risks reaching a completely water-FREE solid core for anything that
    # isn't a hollow shell (a hollow-vs-solid mixup this exact padding once
    # caused: on a solid single-domain test protein, letting the window
    # reach r~0 -- zero water near a solid core's mass-weighted centroid --
    # made np.argmin trivially latch onto r~0 as the "minimum", producing a
    # near-zero boundary that classified essentially all solvent as
    # exterior. See the outer-plateau sanity check below, which now catches
    # this case explicitly instead of silently returning it.
    search_lo = float(np.percentile(radii_per_frame, 5))
    search_hi = float(np.percentile(radii_per_frame, 99)) + 20.0

    edges = np.arange(0.0, search_hi + density_bin_width, density_bin_width)
    counts, _ = np.histogram(all_water_radial, bins=edges)
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    density = counts / (shell_vol * n_calib)
    centers = 0.5 * (edges[1:] + edges[:-1])

    # Light smoothing (~2 A window) to reduce per-bin counting noise before
    # locating the minimum -- the trough itself is a shell-thickness-scale
    # feature (tens of A wide), not a single sharp bin.
    smooth_window = max(1, int(round(2.0 / density_bin_width)))
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        density_smooth = np.convolve(density, kernel, mode="same")
    else:
        density_smooth = density

    in_window = np.where((centers >= search_lo) & (centers <= search_hi))[0]
    if in_window.size == 0:
        raise ValueError(
            "Density calibration search window is empty -- check "
            "--n-shell-calibration-frames and the protein/water selections."
        )
    min_idx = in_window[np.argmin(density_smooth[in_window])]
    r_outer = float(centers[min_idx])
    trough_density = float(density_smooth[min_idx])

    # Sanity check: a genuine shell-to-bulk transition means density at the
    # OUTER edge of the search window (well past the most protruding
    # residues, should be true bulk) is meaningfully higher than the
    # trough. If it isn't, the "minimum" found is not a real dip-then-
    # recover feature -- e.g. it degenerated to the water-starved edge of
    # a solid, non-hollow structure, or the calibration is too noisy/short
    # to resolve one. Don't silently return a bad boundary.
    outer_ref_n = max(3, in_window.size // 5)
    outer_bulk_density = float(np.median(density_smooth[in_window[-outer_ref_n:]]))
    if trough_density > 0.5 * outer_bulk_density:
        logger.warning(
            "[shell calibration] density at the candidate outer boundary "
            "(%.2f A, density=%.4g /A^3) is not clearly a trough relative to "
            "the outer-window bulk density (%.4g /A^3) -- this usually means "
            "the structure doesn't have a genuine hollow-shell water-density "
            "dip (e.g. a solid, non-hollow test system), or the calibration "
            "is too short/noisy to resolve one. The resulting boundary is "
            "likely wrong; check the saved density plot before trusting "
            "this run's classification, and consider more calibration "
            "frames or verifying this system is actually a hollow capsid.",
            r_outer, trough_density, outer_bulk_density,
        )

    # Second sanity check: RECOVERY SPEED. A genuine hollow-shell trough
    # should snap back close to bulk density within roughly one solvation
    # shell's width past it (~15 A) -- the check above alone can still be
    # fooled by a shallow local dip on an otherwise steadily, slowly rising
    # profile (e.g. a solid, non-hollow structure's core-to-surface
    # gradient), since "less than half of eventual bulk" doesn't require
    # getting there quickly. If density is still far from bulk shortly past
    # the trough, the profile is climbing gradually, not recovering from a
    # real shell -- same underlying conclusion as above, flagged
    # separately because it catches cases the first check doesn't.
    recovery_r = r_outer + 15.0
    recovery_candidates = in_window[centers[in_window] >= recovery_r]
    if recovery_candidates.size > 0:
        recovery_density = float(density_smooth[recovery_candidates[0]])
        recovery_frac = recovery_density / outer_bulk_density if outer_bulk_density > 0 else 0.0
        if recovery_frac < 0.7:
            logger.warning(
                "[shell calibration] water density only recovers to %.0f%% of "
                "outer bulk by %.1f A past the candidate outer boundary (%.2f A) "
                "-- a genuine hollow shell should snap back close to bulk within "
                "about one solvation shell's width. This profile looks like a "
                "slow, gradual core-to-surface rise (typical of a solid, non-"
                "hollow structure) rather than a real shell-to-bulk transition. "
                "Check the saved density plot; the outer boundary is likely "
                "placed too far inward to be trustworthy.",
                100.0 * recovery_frac, 15.0, r_outer,
            )

    # Inner boundary: scan from the trough back toward the lumen side and
    # take the point where density has climbed halfway from the trough
    # value back to the lumen's own (roughly constant, if solvated) value.
    # Falls back to a fixed offset if the lumen side isn't well-resolved
    # (e.g. too few bins inside the trough within the search window).
    inner_side = in_window[in_window <= min_idx]
    if inner_side.size >= 3:
        lumen_ref_n = max(3, inner_side.size // 4)
        lumen_density = float(np.median(density_smooth[inner_side[:lumen_ref_n]]))
        threshold = trough_density + 0.5 * (lumen_density - trough_density)
        below = density_smooth[inner_side] <= threshold
        r_inner = float(centers[inner_side[below][-1]]) if below.any() else float(centers[inner_side[0]])
    else:
        r_inner = max(0.0, r_outer - 40.0)

    if rank == 0:
        logger.info(
            "[shell calibration, density-based] %d frames sampled (of %d total) -- "
            "search window [%.1f, %.1f] A, density minimum (outer boundary) at "
            "%.2f A (trough=%.4g /A^3, outer bulk=%.4g /A^3), inner boundary at %.2f A",
            len(calib_indices), n_frames_all, search_lo, search_hi, r_outer,
            trough_density, outer_bulk_density, r_inner,
        )
    return r_inner, r_outer, res_radial_mean, (centers, density_smooth)


def classify_radial_zones(solvent_com, centroid, r_inner_cut, r_outer_cut):
    """Per-solvent-molecule classification relative to the calibrated,
    margin-padded shell boundary. Returns (exterior_mask, interior_mask,
    radial_distance) -- molecules in neither mask are the "ambiguous"
    (shell-embedded) population, deliberately excluded from both totals.
    """
    radial = np.linalg.norm(solvent_com - centroid, axis=1)
    exterior_mask = radial > r_outer_cut
    interior_mask = radial < r_inner_cut
    return exterior_mask, interior_mask, radial


# --------------------------------------------------------------------------
# Hybrid-36 encoding + a minimal standalone PDB writer.
#
# A full AAV capsid (~240,000 atoms) exceeds the legacy PDB format's
# 99999-atom decimal serial-number field (columns 7-11) -- and potentially
# a chain's residue count could exceed the 9999 resSeq field (columns
# 23-26) too. Hybrid-36 (Grosse-Kunstleve et al.; used by PyMOL, ChimeraX,
# Phenix/CCTBX) extends both fields into base-36 once they overflow, while
# leaving anything that fits in plain decimal untouched.
# --------------------------------------------------------------------------
_HY36_UPPER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_HY36_LOWER = "0123456789abcdefghijklmnopqrstuvwxyz"


def _hy36_encode_pure(digits, value):
    if value == 0:
        return digits[0]
    n = len(digits)
    out = []
    while value:
        value, rem = divmod(value, n)
        out.append(digits[rem])
    return "".join(reversed(out))


def hy36encode(width, value):
    """Hybrid-36 encode an integer into a fixed-width PDB field.

    width=5 for the atom serial field (plain-decimal capacity 99999),
    width=4 for resSeq (plain-decimal capacity 9999).
    """
    i = value
    lower_bound = 1 - 10 ** (width - 1)
    if i < lower_bound:
        raise ValueError(f"value {value} too small for hybrid-36 width {width}")
    if i < 10 ** width:
        return ("%" + str(width) + "d") % i
    i -= 10 ** width
    if i < 26 * 36 ** (width - 1):
        i += 10 * 36 ** (width - 1)
        return _hy36_encode_pure(_HY36_UPPER, i)
    i -= 26 * 36 ** (width - 1)
    if i < 26 * 36 ** (width - 1):
        i += 10 * 36 ** (width - 1)
        return _hy36_encode_pure(_HY36_LOWER, i)
    raise ValueError(f"value {value} too large for hybrid-36 width {width}")


def _pdb_atom_name_field(name, element):
    """4-character atom-name field, PDB's standard justification rule:
    left-justified from column 13 for 4-char names or 2-char elements,
    otherwise shifted one space right (element symbol lands on column 14)."""
    name = name.strip()
    if len(name) >= 4:
        return name[:4]
    if len(element.strip()) >= 2:
        return name.ljust(4)[:4]
    return (" " + name).ljust(4)[:4]


_PDB_CHAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def chain_index_to_pdb_id(chain_idx):
    """Map a 0-based chain index to a single legal PDB chain-ID character
    (A-Z, a-z, 0-9 -- 62 total). Wraps past 62 distinct chains."""
    return _PDB_CHAIN_ALPHABET[chain_idx % len(_PDB_CHAIN_ALPHABET)]


def write_pdb_hybrid36(atomgroup, tempfactors, out_path, logger, box_dimensions=None,
                        chain_local_idx=None, remark_lines=None):
    """Write a minimal ATOM-record PDB with hybrid-36 serial/resSeq fields.

    Bypasses MDAnalysis's PDBWriter entirely -- this is the one place a
    whole-capsid run (~240,000 atoms) would otherwise hit the format's
    99999-atom ceiling. `tempfactors` is a per-atom array (same order as
    atomgroup) written into the B-factor column.

    `chain_local_idx`, if given, is a per-atom 0-based chain index (same
    order as atomgroup, e.g. protein_chain_meta["local_idx"]) used to
    assign PDB chain-ID letters directly, INSTEAD of atomgroup.chainIDs/
    segids. This matters: those topology attributes are not always
    reliable for a multi-chain assembly (segid/chainID silently merged two
    physically separate capsid monomers into one group on a real dataset,
    which also broke every downstream tool that groups PDB atoms by
    (chain, resid) -- e.g. two different monomers' residue 457 would both
    be written as chain 'S', collapsing them together). Prefer passing the
    same chain grouping already used for chain-level PBC realignment
    (built from resid_reset_chain_key when usable) so the PDB's chain
    labeling is guaranteed self-consistent with the rest of the pipeline.

    `remark_lines`, if given, replaces the default REMARK header (which
    describes THIS module's own Gamma_23 B-factor convention) with a
    caller-supplied list of strings -- this function is reused as-is by
    other scripts (e.g. pic_symmetry_regions.py) whose B-factor column
    means something else entirely, and the default text would otherwise
    misdescribe their output.
    """
    names = atomgroup.names
    resnames = atomgroup.resnames
    resids = atomgroup.resids
    positions = atomgroup.positions
    elements = [guess_element(n) for n in names]

    if chain_local_idx is not None:
        n_chains = int(chain_local_idx.max()) + 1 if len(chain_local_idx) else 0
        if n_chains > len(_PDB_CHAIN_ALPHABET):
            logger.warning(
                "%d distinct chains exceeds the %d-character PDB chain-ID "
                "alphabet (A-Z, a-z, 0-9) -- chain letters will repeat, so "
                "tools that group atoms by (chain, resid) may still collapse "
                "some chains together. mmCIF has no such limit if this "
                "matters for downstream analysis.", n_chains, len(_PDB_CHAIN_ALPHABET),
            )
        chain_ids = [chain_index_to_pdb_id(int(c)) for c in chain_local_idx]
    elif hasattr(atomgroup, "chainIDs"):
        chain_ids = list(atomgroup.chainIDs)
        chain_ids = [(str(c).strip()[:1] or "A") for c in chain_ids]
    elif hasattr(atomgroup, "segids"):
        chain_ids = list(atomgroup.segids)
        chain_ids = [(str(c).strip()[:1] or "A") for c in chain_ids]
    else:
        chain_ids = ["A"] * len(atomgroup)

    n_atoms = len(atomgroup)
    if n_atoms > 99999:
        logger.info(
            "%d atoms exceeds the classic PDB 99999-atom limit -- writing "
            "hybrid-36 serial numbers (read natively by PyMOL/ChimeraX/"
            "Phenix; tools that only understand plain-decimal PDB will "
            "misparse rows beyond atom 99999, same as they would for any "
            "oversized PDB file).", n_atoms,
        )
    max_resid = int(resids.max()) if n_atoms else 0
    if max_resid > 9999:
        logger.info(
            "max resid %d exceeds the classic PDB 9999 resSeq limit -- "
            "writing hybrid-36 resSeq numbers.", max_resid,
        )

    tf_clipped = np.clip(np.nan_to_num(tempfactors, nan=0.0), -999.99, 999.99)
    if not np.array_equal(tf_clipped, np.nan_to_num(tempfactors, nan=0.0)):
        logger.warning("Some B-factor values were clipped to +/-999.99 to fit the "
                        "PDB tempFactor field width.")

    if remark_lines is None:
        remark_lines = [
            "generated by pic_calculation_exterior.py",
            "B-factor column = per-residue Gamma_23 (EXTERIOR solvent only)",
        ]

    with open(out_path, "w") as fh:
        for line in remark_lines:
            fh.write(f"REMARK   {line}\n")
        fh.write("REMARK   atom serial / resSeq use hybrid-36 numbering above 99999 / 9999\n")
        if box_dimensions is not None:
            a, b, c, alpha, beta, gamma = box_dimensions
            fh.write("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n"
                      % (a, b, c, alpha, beta, gamma))
        for i in range(n_atoms):
            name_field = _pdb_atom_name_field(names[i], elements[i])
            serial = hy36encode(5, i + 1)
            resseq = hy36encode(4, int(resids[i]))
            x, y, z = positions[i]
            fh.write(
                "ATOM  %5s %-4s %-3.3s %1s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n"
                % (serial, name_field, resnames[i], chain_ids[i], resseq,
                   x, y, z, 1.00, tf_clipped[i], elements[i])
            )
        fh.write("END\n")


class RunningStats:
    """Accumulates sum / sum-of-squares / finite-count for an array of fixed
    shape across frames, without ever storing per-frame data."""

    def __init__(self, shape):
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sumsq = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape, dtype=np.float64)

    def update(self, values):
        finite = np.isfinite(values)
        self.sum += np.where(finite, values, 0.0)
        self.sumsq += np.where(finite, values * values, 0.0)
        self.count += finite


class RunningScalar:
    """Simple running sum + count for a plain scalar (e.g. per-frame
    exterior/interior/ambiguous molecule counts), for a mean-over-frames
    diagnostic at the end -- lighter weight than RunningStats since these
    are never NaN and don't need a fixed shape."""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    def mean(self):
        return self.total / self.count if self.count > 0 else float("nan")


# ==========================================================================
# MPI driver
# ==========================================================================

def setup_logging(rank):
    logger = logging.getLogger("pic_exterior")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        f"%(asctime)s [rank {rank:>3}] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gro", required=True, help="Topology/coordinate file (md.gro)")
    p.add_argument("--xtc", required=True, help="Trajectory file (md_noPBC.xtc)")
    p.add_argument("--excipient", required=True, choices=["NA", "CL", "SUC"],
                    help="Residue name of the excipient/cosolvent of interest")
    p.add_argument("--radius", required=True, type=float,
                    help="Radius of selection R_max in Angstrom: the max "
                         "distance from each residue's surface to bin out "
                         "to. Must comfortably exceed the expected r* "
                         "plateau (15-20 A is typical) but need not "
                         "approach the box edge.")
    p.add_argument("--force-rstar", type=float, default=None,
                    help="Skip per-residue r* auto-detection (find_rstar) and "
                         "instead report Gamma_23 at this fixed r (Angstrom) "
                         "for every residue AND the whole-molecule curve. "
                         "Snapped to the nearest --bin-width bin; must be "
                         "<= --radius. Use this for run-to-run consistency "
                         "(e.g. comparing residues/systems at the exact same "
                         "r) instead of letting each residue/system pick its "
                         "own auto-detected plateau.")
    p.add_argument("--charge", type=float, default=0.0,
                    help="Starting (net protein formal) charge, used only "
                         "for the optional charge-balance diagnostic when "
                         "--excipient is NA or CL. Leave at 0 for SUC or if "
                         "not needed.")
    p.add_argument("--water-resname", default="SOL",
                    help="Residue name for water (default: SOL)")
    p.add_argument("--protein-selection", default="protein",
                    help="MDAnalysis selection string for the protein (default: 'protein')")
    p.add_argument("--bin-width", type=float, default=0.1,
                    help="Histogram bin width in Angstrom (default 0.1 A, "
                         "matching the source procedure)")
    p.add_argument("--output-dir", default="pic_results_exterior",
                    help="Directory for output files (default: pic_results_exterior)")
    p.add_argument("--stride", type=int, default=1,
                    help="Process every Nth frame (default 1 = all frames). "
                         "Useful for a quick test run before the full job.")
    p.add_argument("--max-frames", type=int, default=None,
                    help="Cap total frames processed (after stride), for "
                         "quick sanity-check runs. Default: all frames.")
    p.add_argument("--reference-frame", type=int, default=0,
                    help="Frame index whose coordinates are used for the "
                         "B-factor-colored PDB output (default: 0)")
    p.add_argument("--log-every", type=int, default=200,
                    help="Log progress every N local frames per rank (default 200)")
    p.add_argument("--n-checkpoints", type=int, default=20,
                    help="Number of synchronized cluster-wide aggregate "
                         "progress/ETA reports over the course of the run "
                         "(default 20, i.e. roughly every 5%%).")
    # EXTERIOR-ONLY options
    p.add_argument("--shell-margin", type=float, default=2.0,
                    help="Angstrom margin added beyond the calibrated shell "
                         "radii (the water density minimum/maximum -- see "
                         "calibrate_shell_geometry) before classifying "
                         "solvent as confidently interior/exterior (default "
                         "2.0 A -- smaller than earlier versions since the "
                         "density-based calibration already places the "
                         "boundary close to the true protein surface, unlike "
                         "the old percentile-of-residue-position approach). "
                         "Solvent within the margin band is excluded from "
                         "both categories.")
    p.add_argument("--density-bin-width", type=float, default=2.0,
                    help="Bin width in Angstrom for the water radial number-"
                         "density profile used to locate the shell boundary "
                         "(default 2.0 A -- coarser than --bin-width since "
                         "this profile spans hundreds of A and the boundary "
                         "is a shell-thickness-scale feature, not a sharp "
                         "single-bin transition).")
    p.add_argument("--n-shell-calibration-frames", type=int, default=20,
                    help="Number of frames, spread evenly across the whole "
                         "trajectory, used to calibrate the shell's inner/"
                         "outer radii before the main loop (default 20). "
                         "Every rank samples the same deterministic frames.")
    return p.parse_args()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    logger = setup_logging(rank)

    args = parse_args()
    t_start = time.time()

    if rank == 0:
        logger.info("=" * 70)
        logger.info("PIC (Gamma_23) calculation, per-residue, EXTERIOR SOLVENT ONLY")
        logger.info("gro=%s xtc=%s excipient=%s radius=%.2f A charge=%.3f",
                     args.gro, args.xtc, args.excipient, args.radius, args.charge)
        logger.info("shell-margin=%.1f A  density-bin-width=%.1f A  calib-frames=%d",
                     args.shell_margin, args.density_bin_width, args.n_shell_calibration_frames)
        logger.info("MPI ranks: %d%s", size, "" if _HAS_REAL_MPI else " (no MPI runtime -- serial fallback)")
        for f in (args.gro, args.xtc):
            if not os.path.isfile(f):
                logger.error("Input file not found: %s", f)
                comm.Abort(1)
        if args.force_rstar is not None:
            if args.force_rstar <= 0 or args.force_rstar > args.radius:
                logger.error(
                    "--force-rstar %.3f A must be > 0 and <= --radius (%.3f A)",
                    args.force_rstar, args.radius,
                )
                comm.Abort(1)
            logger.info("r* FORCED to %.2f A for every residue and the whole-molecule "
                        "curve (auto-detection disabled)", args.force_rstar)
        os.makedirs(args.output_dir, exist_ok=True)
    comm.Barrier()

    # ---- topology / selections (every rank does this independently) ----
    u = mda.Universe(args.gro, args.xtc)
    protein = u.select_atoms(args.protein_selection)
    water = u.select_atoms(f"resname {args.water_resname}")
    excipient = u.select_atoms(f"resname {args.excipient}")

    for ag, label in [(protein, "protein"), (water, "water"), (excipient, "excipient")]:
        validate_masses(ag, label, logger)

    protein_vdw = get_protein_vdw_radii(protein, logger)
    protein_masses = protein.masses
    water_masses = water.masses
    excipient_masses = excipient.masses

    n_res = protein.residues.n_residues
    res_resids = protein.residues.resids
    res_resnames = protein.residues.resnames
    n1_topology_total = water.residues.n_residues
    n3_topology_total = excipient.residues.n_residues

    protein_meta = build_group_meta(protein.resindices, n_res, "protein", logger)
    water_meta = build_group_meta(water.resindices, n1_topology_total, "water", logger)
    excipient_meta = build_group_meta(excipient.resindices, n3_topology_total, "excipient", logger)

    # Multiple candidate signals are computed and cross-logged: segid/
    # chainID metadata is NOT always reliable for this -- on a real capsid
    # run it silently merged two physically separate monomers into one
    # group (both still reported a plausible-looking distinct-group
    # count), which rigid_align_chains then could not correctly separate.
    # resid-reset is an independent, purely-topological signal that
    # doesn't depend on that metadata being correct, and is preferred
    # whenever it's usable.
    chain_candidates = [("segindices", protein.segindices)]
    if hasattr(protein, "chainIDs"):
        chain_candidates.append(("chainIDs", protein.chainIDs))
    resid_reset_key = resid_reset_chain_key(protein)
    if resid_reset_key is not None:
        chain_candidates.append(("resid-reset", resid_reset_key))

    if rank == 0:
        for cand_label, cand_key in chain_candidates:
            uniq, group_sizes = np.unique(cand_key, return_counts=True)
            logger.info(
                "[protein] chain-grouping candidate '%s': %d distinct groups, "
                "member counts min=%d max=%d mean=%.1f (uniform sizes are a "
                "good sign for a multi-copy assembly like a capsid)",
                cand_label, uniq.size, group_sizes.min(), group_sizes.max(), group_sizes.mean(),
            )

    resid_reset_entry = next((c for c in chain_candidates if c[0] == "resid-reset"), None)
    if resid_reset_entry is not None:
        chain_label, chain_key = resid_reset_entry
    else:
        chain_label, chain_key = max(chain_candidates, key=lambda kv: np.unique(kv[1]).size)
    n_chains = np.unique(chain_key).size
    if rank == 0:
        logger.info("[protein] using '%s' for chain-level PBC realignment (%d distinct chains/segments)",
                    chain_label, n_chains)
        if n_chains == 1:
            logger.warning(
                "Only 1 chain/segment detected for the protein selection -- "
                "if this is a multi-copy assembly (e.g. a capsid), chain-level "
                "PBC realignment cannot separate copies and will be a no-op. "
                "Check that the topology carries per-chain segid/chainID info, "
                "or that resid numbering resets per chain."
            )
    protein_chain_meta = build_group_meta(chain_key, n_chains, "protein-chain", logger)

    if rank == 0:
        logger.info("protein atoms=%d residues=%d  water molecules=%d (topology total)  "
                     "%s molecules=%d (topology total)",
                     protein.n_atoms, n_res, n1_topology_total, args.excipient, n3_topology_total)
        if n3_topology_total == 0:
            logger.error("Zero %s residues found -- check --excipient / topology naming.",
                          args.excipient)
            comm.Abort(1)

    # ---- EXTERIOR-ONLY: calibrate the capsid shell's radial extent once,
    # before the main loop, identically on every rank ----
    r_inner, r_outer, res_radial_mean, density_profile = calibrate_shell_geometry(
        u, protein, protein_meta, protein_chain_meta, protein_masses,
        water, water_meta, water_masses, n_res,
        args.n_shell_calibration_frames, args.density_bin_width, logger, rank,
    )
    r_inner_cut = r_inner - args.shell_margin
    r_outer_cut = r_outer + args.shell_margin
    if r_inner_cut >= r_outer_cut:
        logger.error(
            "--shell-margin (%.1f A) is larger than the calibrated shell "
            "thickness (inner=%.2f, outer=%.2f A) -- every solvent molecule "
            "would be classified ambiguous. Reduce --shell-margin.",
            args.shell_margin, r_inner, r_outer,
        )
        comm.Abort(1)
    if rank == 0:
        logger.info("Exterior classification cutoff: radial distance > %.2f A "
                     "(outer radius %.2f + margin %.2f)", r_outer_cut, r_outer, args.shell_margin)
        logger.info("Interior classification cutoff: radial distance < %.2f A "
                     "(inner radius %.2f - margin %.2f)", r_inner_cut, r_inner, args.shell_margin)

    # Per-residue outer/inner "facing" diagnostic: simple median split of
    # each residue's own mean radial distance from the calibration frames.
    facing_median = np.median(res_radial_mean)
    res_facing_outer = res_radial_mean > facing_median
    if rank == 0:
        logger.info("Per-residue facing (median split at %.2f A): %d outer-facing, %d inner-facing",
                    facing_median, int(res_facing_outer.sum()), int((~res_facing_outer).sum()))

    n_bins = int(round(args.radius / args.bin_width))
    bin_edges = np.linspace(0.0, n_bins * args.bin_width, n_bins + 1)
    r_values = bin_edges[1:]

    # ---- frame range for this rank ----
    n_frames_all = len(u.trajectory)
    frame_indices = list(range(0, n_frames_all, args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    n_frames = len(frame_indices)

    if rank == 0:
        logger.info("trajectory frames: %d total, %d selected (stride=%d, max_frames=%s)",
                     n_frames_all, n_frames, args.stride, args.max_frames)
        frames_per_rank = n_frames / size if size > 0 else n_frames
        if size > 1 and frames_per_rank < MIN_USEFUL_FRAMES_PER_RANK:
            suggested = max(1, -(-n_frames // MIN_USEFUL_FRAMES_PER_RANK))  # ceil div
            logger.warning(
                "Only %.1f frames/rank at %d ranks for this %d-frame run -- "
                "per-rank fixed cost (Universe/topology load, MPI startup, "
                "shell calibration) will dominate over actual compute "
                "(~%.2fs/frame). For a run this size, --ntasks=%d (or fewer) "
                "would likely finish just as fast while using less of the "
                "allocation.",
                frames_per_rank, size, n_frames, SECONDS_PER_FRAME_ESTIMATE, suggested,
            )

    start, stop = split_frame_range(n_frames, rank, size)
    my_frames = frame_indices[start:stop]
    logger.info("assigned %d frames (local index %d:%d)", len(my_frames), start, stop)

    # ---- accumulators (fixed size, independent of trajectory length) ----
    gamma_res_stats = RunningStats((n_res, n_bins))
    whole_stats = RunningStats((n_bins,))
    n3_r_whole_stats = RunningStats((n_bins,))  # for the charge diagnostic mean
    n_water_ext_stats = RunningScalar()
    n_water_int_stats = RunningScalar()
    n_exc_ext_stats = RunningScalar()
    n_exc_int_stats = RunningScalar()
    n_local = len(my_frames)

    # Aggregate (cluster-wide) progress via periodic BLOCKING Allreduce.
    #
    # CRITICAL CORRECTNESS REQUIREMENT: every rank must call Allreduce the
    # same number of times, in the same order -- see split_frame_range /
    # max_local below, unchanged from pic_calculation.py.
    max_local = comm.allreduce(n_local, op=MPI.MAX)
    N_CHECKPOINTS = max(1, args.n_checkpoints)
    if max_local > 0:
        checkpoint_iters_set = set(
            max(0, int(round((k + 1) * max_local / N_CHECKPOINTS)) - 1)
            for k in range(N_CHECKPOINTS)
        )
        checkpoint_iters_set.add(max_local - 1)  # always report the final round
    else:
        checkpoint_iters_set = set()
    loop_t0 = time.time()

    for i in range(max_local):
        if i < n_local:
            fidx = my_frames[i]
            u.trajectory[fidx]
            box = u.dimensions  # may fluctuate frame-to-frame under NPT

            # Protein positions used AS-IS -- see the matching comment in
            # calibrate_shell_geometry for why unwrap/chain-realignment is
            # deliberately NOT applied here (it mis-shifted already-correct
            # chains on a -pbc-whole trajectory without fixing the real
            # issue, which is solvent-side, not protein-side).
            protein_pos = protein.positions
            res_com = mass_weighted_com(protein_pos, protein_masses, protein_meta)
            res_radius = residue_radius(
                protein_pos, res_com, protein_meta, protein_vdw,
                logger=logger if i == 0 else None,
            )
            capsid_centroid = res_com.mean(axis=0)

            # Water/excipient DO need re-imaging to their nearest periodic
            # copy relative to the centroid -- see calibrate_shell_geometry
            # for the direct verification (205/517 -> 517/517 residues
            # gaining a real water neighbor on the worst-affected chain).
            water_pos = water.positions
            water_com_all = mass_weighted_com(water_pos, water_masses, water_meta)
            exc_pos = excipient.positions
            exc_com_all = mass_weighted_com(exc_pos, excipient_masses, excipient_meta)
            water_com_all = capsid_centroid + minimize_vectors(water_com_all - capsid_centroid, box)
            exc_com_all = capsid_centroid + minimize_vectors(exc_com_all - capsid_centroid, box)

            # EXTERIOR-ONLY: classify and filter before anything downstream
            # ever sees these arrays.
            water_ext_mask, water_int_mask, _ = classify_radial_zones(
                water_com_all, capsid_centroid, r_inner_cut, r_outer_cut)
            exc_ext_mask, exc_int_mask, _ = classify_radial_zones(
                exc_com_all, capsid_centroid, r_inner_cut, r_outer_cut)

            water_com = water_com_all[water_ext_mask]
            exc_com = exc_com_all[exc_ext_mask]
            n1_total = int(water_ext_mask.sum())
            n3_total = int(exc_ext_mask.sum())

            n_water_ext_stats.update(n1_total)
            n_water_int_stats.update(int(water_int_mask.sum()))
            n_exc_ext_stats.update(n3_total)
            n_exc_int_stats.update(int(exc_int_mask.sum()))

            w_hist, w_mol_min = frame_residue_histograms(
                water_com, res_com, res_radius, n_res, n_bins, args.bin_width, args.radius)
            e_hist, e_mol_min = frame_residue_histograms(
                exc_com, res_com, res_radius, n_res, n_bins, args.bin_width, args.radius)

            n1_r_res = np.cumsum(w_hist, axis=1)
            n3_r_res = np.cumsum(e_hist, axis=1)
            gamma_res = gamma_from_cumulative_counts(n1_r_res, n3_r_res, n1_total, n3_total)
            gamma_res_stats.update(gamma_res)

            # whole-molecule validation curve, same exterior-only inputs.
            w_hist_whole, _ = np.histogram(w_mol_min[np.isfinite(w_mol_min)], bins=bin_edges)
            e_hist_whole, _ = np.histogram(e_mol_min[np.isfinite(e_mol_min)], bins=bin_edges)
            n1_r_whole = np.cumsum(w_hist_whole)
            n3_r_whole = np.cumsum(e_hist_whole)
            gamma_whole = gamma_from_cumulative_counts(n1_r_whole, n3_r_whole, n1_total, n3_total)
            whole_stats.update(gamma_whole)
            n3_r_whole_stats.update(n3_r_whole.astype(float))

            if (i + 1) % args.log_every == 0 or i == n_local - 1:
                elapsed_loop = time.time() - loop_t0
                rate = (i + 1) / elapsed_loop if elapsed_loop > 0 else 0.0
                remaining = n_local - (i + 1)
                eta_local = remaining / rate if rate > 0 else None
                logger.info(
                    "frame %d/%d (%.1f%%, global frame %d) | rate=%.2f fr/s | "
                    "local ETA=%s | exterior water=%d exc=%d",
                    i + 1, n_local, 100.0 * (i + 1) / n_local, fidx,
                    rate, format_eta(eta_local), n1_total, n3_total,
                )

        if i in checkpoint_iters_set:
            my_contribution = min(i + 1, n_local)
            total_done = comm.allreduce(my_contribution, op=MPI.SUM)
            if rank == 0:
                frac = total_done / n_frames if n_frames > 0 else 0.0
                elapsed_total = time.time() - t_start
                rate_total = total_done / elapsed_total if elapsed_total > 0 else 0.0
                eta_global = (n_frames - total_done) / rate_total if rate_total > 0 else None
                logger.info(
                    "[AGGREGATE] %d/%d frames complete cluster-wide (%.1f%%) | "
                    "aggregate rate=%.2f fr/s | global ETA=%s | elapsed=%s",
                    total_done, n_frames, 100.0 * frac, rate_total,
                    format_eta(eta_global), format_eta(elapsed_total),
                )

    # ---- reduce accumulators onto rank 0 ----
    gamma_res_sum = comm.reduce(gamma_res_stats.sum, op=MPI.SUM, root=0)
    gamma_res_sumsq = comm.reduce(gamma_res_stats.sumsq, op=MPI.SUM, root=0)
    gamma_res_count = comm.reduce(gamma_res_stats.count, op=MPI.SUM, root=0)

    gamma_whole_sum = comm.reduce(whole_stats.sum, op=MPI.SUM, root=0)
    gamma_whole_sumsq = comm.reduce(whole_stats.sumsq, op=MPI.SUM, root=0)
    gamma_whole_count = comm.reduce(whole_stats.count, op=MPI.SUM, root=0)

    n3_r_whole_sum = comm.reduce(n3_r_whole_stats.sum, op=MPI.SUM, root=0)
    n3_r_whole_count = comm.reduce(n3_r_whole_stats.count, op=MPI.SUM, root=0)

    # Scalar diagnostics: reduce (local_total, local_count) pairs so rank 0
    # can compute the true cluster-wide mean, not a mean-of-per-rank-means.
    water_ext_total = comm.reduce(n_water_ext_stats.total, op=MPI.SUM, root=0)
    water_ext_count = comm.reduce(n_water_ext_stats.count, op=MPI.SUM, root=0)
    water_int_total = comm.reduce(n_water_int_stats.total, op=MPI.SUM, root=0)
    exc_ext_total = comm.reduce(n_exc_ext_stats.total, op=MPI.SUM, root=0)
    exc_ext_count = comm.reduce(n_exc_ext_stats.count, op=MPI.SUM, root=0)
    exc_int_total = comm.reduce(n_exc_int_stats.total, op=MPI.SUM, root=0)

    if rank != 0:
        return

    def mean_sem_from_parts(s, ssq, cnt):
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = s / cnt
            var = np.maximum(ssq / cnt - mean * mean, 0.0)
            sem = np.sqrt(var / np.maximum(cnt, 1.0))
        mean = np.where(cnt > 0, mean, np.nan)
        sem = np.where(cnt > 0, sem, np.nan)
        return mean, sem

    gamma_res_mean, gamma_res_sem = mean_sem_from_parts(gamma_res_sum, gamma_res_sumsq, gamma_res_count)
    gamma_whole_mean, gamma_whole_sem = mean_sem_from_parts(gamma_whole_sum, gamma_whole_sumsq, gamma_whole_count)
    n3_r_whole_mean = np.where(n3_r_whole_count > 0, n3_r_whole_sum / np.maximum(n3_r_whole_count, 1.0), np.nan)

    logger.info("aggregated per-residue Gamma_23(r,t): shape %s over %d rank-frames",
                gamma_res_mean.shape, int(gamma_res_count.max()) if gamma_res_count.size else 0)

    mean_water_ext = water_ext_total / water_ext_count if water_ext_count else float("nan")
    mean_water_int = water_int_total / water_ext_count if water_ext_count else float("nan")
    mean_exc_ext = exc_ext_total / exc_ext_count if exc_ext_count else float("nan")
    mean_exc_int = exc_int_total / exc_ext_count if exc_ext_count else float("nan")
    logger.info(
        "Mean per-frame classification: water exterior=%.0f interior=%.0f "
        "(of %d topology total) | %s exterior=%.0f interior=%.0f (of %d topology total)",
        mean_water_ext, mean_water_int, n1_topology_total,
        args.excipient, mean_exc_ext, mean_exc_int, n3_topology_total,
    )
    if mean_exc_ext < 1.0:
        logger.warning(
            "Mean exterior %s count is <1 per frame -- check --shell-margin/"
            "percentiles and the shell calibration log above; Gamma_23 is "
            "meaningless with an empty exterior reservoir.", args.excipient
        )

    # ---- r* and final Gamma_23 per residue ----
    # --force-rstar pins ONE bin index, used for every residue and the
    # whole-molecule curve alike, instead of each independently calling
    # find_rstar -- for run-to-run / residue-to-residue consistency at a
    # known r, rather than each auto-detecting its own plateau.
    forced_idx = (int(np.argmin(np.abs(r_values - args.force_rstar)))
                  if args.force_rstar is not None else None)

    rstar_res = np.full(n_res, np.nan)
    gamma23_res = np.full(n_res, np.nan)
    gamma23_res_sem = np.full(n_res, np.nan)
    for ridx in range(n_res):
        curve = gamma_res_mean[ridx]
        if not np.any(np.isfinite(curve)):
            continue
        if forced_idx is not None:
            rstar, rstar_idx = r_values[forced_idx], forced_idx
        else:
            rstar, rstar_idx = find_rstar(r_values, curve)
        rstar_res[ridx] = rstar
        gamma23_res[ridx] = curve[rstar_idx]
        gamma23_res_sem[ridx] = gamma_res_sem[ridx, rstar_idx]

    if forced_idx is not None:
        rstar_whole, rstar_whole_idx = r_values[forced_idx], forced_idx
    else:
        rstar_whole, rstar_whole_idx = find_rstar(r_values, gamma_whole_mean)
    gamma23_whole = gamma_whole_mean[rstar_whole_idx]
    gamma23_whole_sem = gamma_whole_sem[rstar_whole_idx]

    logger.info("%s whole-molecule r* = %.2f A%s",
                "FORCED" if forced_idx is not None else "AUTO-DETECTED",
                rstar_whole,
                "" if forced_idx is not None else " (CONFIRM VISUALLY against the saved plot)")
    logger.info("Whole-molecule Gamma_23 (exterior-only, residue-sphere approximation) = %.4f +/- %.4f",
                gamma23_whole, gamma23_whole_sem)
    finite_outer = np.isfinite(gamma23_res) & res_facing_outer
    finite_inner = np.isfinite(gamma23_res) & ~res_facing_outer
    if finite_outer.any():
        logger.info(
            "Per-residue Gamma_23, OUTER-facing: min=%.4f max=%.4f mean=%.4f (n=%d)",
            np.nanmin(gamma23_res[finite_outer]), np.nanmax(gamma23_res[finite_outer]),
            np.nanmean(gamma23_res[finite_outer]), int(finite_outer.sum()),
        )
    if finite_inner.any():
        logger.info(
            "Per-residue Gamma_23, INNER-facing (expect near-zero/degenerate -- "
            "no access to exterior solvent by construction): min=%.4f max=%.4f "
            "mean=%.4f (n=%d)",
            np.nanmin(gamma23_res[finite_inner]), np.nanmax(gamma23_res[finite_inner]),
            np.nanmean(gamma23_res[finite_inner]), int(finite_inner.sum()),
        )

    # ---- save raw data ----
    payload = {
        "metadata": {
            "gro": args.gro, "xtc": args.xtc, "excipient": args.excipient,
            "radius": args.radius, "bin_width": args.bin_width,
            "starting_charge": args.charge, "water_resname": args.water_resname,
            "n1_topology_total_water": int(n1_topology_total),
            "n3_topology_total_excipient": int(n3_topology_total),
            "mean_exterior_water_per_frame": float(mean_water_ext),
            "mean_interior_water_per_frame": float(mean_water_int),
            "mean_exterior_excipient_per_frame": float(mean_exc_ext),
            "mean_interior_excipient_per_frame": float(mean_exc_int),
            "shell_inner_radius_angstrom": r_inner, "shell_outer_radius_angstrom": r_outer,
            "shell_margin_angstrom": args.shell_margin,
            "shell_inner_cutoff_angstrom": r_inner_cut, "shell_outer_cutoff_angstrom": r_outer_cut,
            "n_residues": int(n_res), "n_frames": int(n_frames), "n_mpi_ranks": size,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method_note": (
                "EXTERIOR-ONLY variant: shell boundary is calibrated from the "
                "water radial number-density profile (minimum = outer "
                "boundary), not a percentile of residue positions -- see "
                "calibrate_shell_geometry docstring for why. n1_total/n3_total "
                "are recomputed each frame as that frame's exterior-classified "
                "(radial distance from capsid centroid > outer boundary + "
                "margin) water/excipient counts, not fixed topology totals. "
                "Interior and shell-ambiguous solvent is excluded entirely. "
                "Per-residue Gamma_23 uses a residue-sphere approximation "
                "(center = residue COM, radius = farthest own atom + its "
                "VdW radius); the local-domain search only ever sees "
                "exterior solvent COMs, so inner-facing residues are "
                "expected to show degenerate near-zero results -- see the "
                "'facing' column."
            ),
        },
        "shell_density_profile": {
            "radial_angstrom": density_profile[0], "density_per_A3": density_profile[1],
        },
        "r_values": r_values,
        "residue": {
            "resids": np.asarray(res_resids), "resnames": np.asarray(res_resnames),
            "gamma_r_mean": gamma_res_mean, "gamma_r_sem": gamma_res_sem,
            "rstar": rstar_res, "gamma_23_final": gamma23_res, "gamma_23_final_sem": gamma23_res_sem,
            "mean_radial_distance_angstrom": res_radial_mean, "facing_outer": res_facing_outer,
        },
        "whole_molecule": {
            "gamma_r_mean": gamma_whole_mean, "gamma_r_sem": gamma_whole_sem,
            "rstar_auto": rstar_whole, "gamma_23_final": gamma23_whole, "gamma_23_final_sem": gamma23_whole_sem,
        },
    }

    charge_profile = build_charge_profile(r_values, n3_r_whole_mean, args.charge, args.excipient)
    if charge_profile is not None:
        payload["charge_profile_r"] = charge_profile[0]
        payload["charge_profile_Qr"] = charge_profile[1]

    pkl_path = os.path.join(args.output_dir, f"pic_{args.excipient}_exterior.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(payload, fh)
    logger.info("saved raw data -> %s", pkl_path)

    csv_res_path = os.path.join(args.output_dir, f"pic_{args.excipient}_exterior_per_residue.csv")
    with open(csv_res_path, "w") as fh:
        fh.write("resid,resname,gamma23,gamma23_sem,rstar_angstrom,"
                 "mean_radial_distance_angstrom,facing\n")
        for ridx in range(n_res):
            facing = "outer" if res_facing_outer[ridx] else "inner"
            fh.write(f"{res_resids[ridx]},{res_resnames[ridx]},"
                     f"{gamma23_res[ridx]:.6g},{gamma23_res_sem[ridx]:.6g},{rstar_res[ridx]:.3g},"
                     f"{res_radial_mean[ridx]:.3g},{facing}\n")
    logger.info("saved per-residue table -> %s", csv_res_path)

    csv_whole_path = os.path.join(args.output_dir, f"pic_{args.excipient}_exterior_vs_r.csv")
    np.savetxt(csv_whole_path, np.column_stack([r_values, gamma_whole_mean, gamma_whole_sem]),
               header="r_angstrom,gamma23_mean,gamma23_sem", delimiter=",", comments="")
    logger.info("saved whole-molecule r-profile -> %s", csv_whole_path)

    # ---- plot: whole-molecule r-profile + per-residue bar (colored by facing) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(r_values, gamma_whole_mean, color="#2b6cb0", lw=1.5, label=r"$\Gamma_{23}(r)$")
    ax1.fill_between(r_values, gamma_whole_mean - gamma_whole_sem, gamma_whole_mean + gamma_whole_sem,
                      color="#2b6cb0", alpha=0.25, label="SEM")
    rstar_label = "forced" if forced_idx is not None else "auto"
    ax1.axvline(rstar_whole, color="#c53030", ls="--", lw=1.2, label=f"{rstar_label} r* = {rstar_whole:.1f} A")
    ax1.axhline(gamma23_whole, color="#2f855a", ls=":", lw=1.0,
                label=f"Gamma_23 = {gamma23_whole:.3f}")
    ax1.set_xlabel("r, distance from nearest residue surface (A)")
    ax1.set_ylabel(r"$\Gamma_{23}(r)$")
    ax1.set_title(f"Whole-molecule, exterior-only (validation): {args.excipient}")
    ax1.legend(frameon=False, fontsize=8)

    gamma23_plot = np.nan_to_num(gamma23_res)
    bar_colors = np.where(res_facing_outer,
                           np.where(gamma23_plot >= 0, "#2f855a", "#c53030"),
                           "#a0aec0")  # grey for inner-facing regardless of sign
    ax2.bar(res_resids, gamma23_plot, color=bar_colors, width=1.0)
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.set_xlabel("residue ID")
    ax2.set_ylabel(r"$\Gamma_{23}$ (per residue, exterior solvent only)")
    ax2.set_title(f"Per-residue Gamma_23: {args.excipient} (grey = inner-facing)")

    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, f"pic_{args.excipient}_exterior_summary.png")
    fig.savefig(plot_path, dpi=150)
    logger.info("saved plot -> %s", plot_path)

    # ---- diagnostic: water density profile with calibrated shell boundary ----
    fig_d, ax_d = plt.subplots(figsize=(7, 5))
    dens_r, dens_v = density_profile
    ax_d.plot(dens_r, dens_v, color="#2b6cb0", lw=1.3, label="water number density")
    ax_d.axvline(r_inner, color="#805ad5", ls="--", lw=1.2, label=f"inner boundary = {r_inner:.1f} A")
    ax_d.axvline(r_outer, color="#c53030", ls="--", lw=1.2, label=f"outer boundary (density min) = {r_outer:.1f} A")
    ax_d.axvline(r_inner_cut, color="#805ad5", ls=":", lw=1.0, alpha=0.7, label="inner cutoff (+margin)")
    ax_d.axvline(r_outer_cut, color="#c53030", ls=":", lw=1.0, alpha=0.7, label="outer cutoff (+margin)")
    ax_d.set_xlabel("radial distance from capsid centroid (A)")
    ax_d.set_ylabel(r"water number density (molecules / $A^3$)")
    ax_d.set_title("Shell boundary calibration: water radial density profile")
    ax_d.legend(frameon=False, fontsize=8)
    fig_d.tight_layout()
    density_plot_path = os.path.join(args.output_dir, f"pic_{args.excipient}_shell_density.png")
    fig_d.savefig(density_plot_path, dpi=150)
    logger.info("saved shell calibration diagnostic plot -> %s", density_plot_path)

    # ---- PDB with B-factor = per-residue Gamma_23 ----
    u.trajectory[args.reference_frame]
    protein_ref = u.select_atoms(args.protein_selection)
    ref_box = u.dimensions
    # DIAGNOSTIC: PBC correction bypassed here too -- writing raw, uncorrected
    # positions -- so this PDB shows exactly what's in the trajectory file
    # itself, with no interference from this script. See the per-frame loop
    # above for the matching bypass and how to revert both.
    per_atom_gamma = gamma23_res[protein_meta["local_idx"]]
    per_atom_gamma = np.where(np.isfinite(per_atom_gamma), per_atom_gamma, 0.0)
    pdb_path = os.path.join(args.output_dir, f"pic_{args.excipient}_exterior_bfactor.pdb")
    write_pdb_hybrid36(protein_ref, per_atom_gamma, pdb_path, logger,
                        box_dimensions=ref_box if ref_box is not None else None,
                        chain_local_idx=protein_chain_meta["local_idx"])
    logger.info("saved B-factor-colored structure (frame %d) -> %s", args.reference_frame, pdb_path)

    elapsed = time.time() - t_start
    logger.info("TOTAL RUNTIME: %.1f s (%.2f frames/s aggregate)",
                 elapsed, n_frames / elapsed if elapsed > 0 else float("nan"))
    logger.info("=" * 70)
    logger.info("RESULT: whole-molecule Gamma_23(%s), exterior-only = %.4f +/- %.4f (r* = %.2f A, %s)",
                 args.excipient, gamma23_whole, gamma23_whole_sem, rstar_whole,
                 "forced" if forced_idx is not None else "auto-detected")
    logger.info("Per-residue breakdown (with facing diagnostic): %s", csv_res_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
