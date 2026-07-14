#!/usr/bin/env python3
"""
plot_symmetry_jitter.py
=======================

Jittered 3-fold/5-fold region plot across every condition in a PIC results
directory, with the whole-capsid Gamma_23 alongside it for context.

Reads, per condition folder:
  pic_symmetry_regions/per_region.csv   -- one row per icosahedral region
                                            (20x 3F + 12x 5F), column `mean`
  pic_SUC_exterior_vs_r.csv             -- whole-molecule Gamma_23(r); the
                                            LAST row is the value at r*
  pic_SUC_exterior.pkl                  -- (only with --normalize) metadata
                                            for the bulk-density normalizer

Layout: one row per condition. Left panel is the whole-capsid Gamma_23 with
error bars; right panel is that condition's 32 region means, jittered (3F
above the row centre, 5F below), with the two group means overlaid as large
dots.

TWO THINGS TO KNOW BEFORE READING THE OUTPUT
--------------------------------------------
1. Gamma_23 is an excess molecule COUNT, so both the values and their spread
   scale with bulk sucrose. Rows at 10-12% sucrose look "wider" than rows at
   1% largely for that reason alone. --normalize divides by each run's own
   bulk mole ratio (n3/n1, from the .pkl metadata) to put every condition on
   a concentration-free scale.

   Both panels use that same single normalizer, so each keeps the quantity
   it started with -- whole-capsid PIC on the left, per-residue PIC on the
   right -- now expressed per unit of bulk sucrose rather than as a raw
   count. --normalize therefore only rescales each row; it never changes
   what a panel is measuring.

   Note this does NOT change the significance of any 3F-vs-5F gap: that is a
   within-condition comparison, so the same factor divides out of both the
   gap and its error.

2. The LEFT panel's error bars are SEMs the pipeline computed by treating
   all frames as independent. They aren't -- sucrose crosses a 15 A shell in
   ~720 ps, so a 10 ns run holds ~14 independent shell-exchange events, not
   ~1000 frames. --corr inflates those SEMs by that factor (default 8.5 =
   sqrt(1001/14)). Pass --corr 1 for the pipeline's raw, over-optimistic SEMs.

   --corr deliberately does NOT apply to the 5F-3F gap table printed at the
   end. That test uses the empirical scatter ACROSS regions (20x 3F, 12x 5F)
   as its error estimate, which already contains whatever frame noise is
   present -- correcting it again would double-count. Any common-mode
   fluctuation of the whole solvent bath also cancels in the 5F-minus-3F
   difference, so it cannot bias the gap. Regions, not residues, are the
   unit of replication here: residues within one region are not independent
   samples, so using the per-residue SEM from overall.csv for this test
   would be pseudo-replication and gives error bars that are far too tight.

Usage
-----
    python plot_symmetry_jitter.py                     # scans its own directory
    python plot_symmetry_jitter.py --normalize
    python plot_symmetry_jitter.py --root /path/to/runs --out fig.png
"""

import argparse
import csv
import os
import pickle
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Condition folder naming: 10ns_aav1ph<PH>salt<MM>suc<PCT>
# pH is written without a decimal point ("73" -> 7.3), so it needs decoding
# rather than a plain float().
# ---------------------------------------------------------------------------
FOLDER_RE = re.compile(r"aav1ph(\d+)salt(\d+)suc(\d+)")

# Formulations from the ranking study; everything else is a DOE corner.
# key = (pH, NaCl mM, sucrose %) -> (label, stability rank, 1 = best)
FORMULATIONS = {
    (6.0, 110, 8):  ("C", 1),
    (7.3, 170, 5):  ("F", 2),
    (6.0, 30, 12):  ("D", 3),
    (7.3, 30, 12):  ("E", 4),
}

TEAL, PLUM, AMBER = "#1f6f5c", "#7a3b66", "#a15c00"
INK, GRID = "#1b2624", "#d8dcda"

DEFAULT_CORR = 8.5    # sqrt(1001 frames / ~14 independent samples)


def decode_ph(raw):
    """'6' -> 6.0, '73' -> 7.3, '8' -> 8.0."""
    return float(raw) if len(raw) == 1 else float(raw[0] + "." + raw[1:])


def load_condition(path):
    """Pull one condition's region means + whole-capsid Gamma_23 off disk.

    Returns None (with a warning) if either required file is missing, so a
    partially-complete results tree still plots what it has.
    """
    name = os.path.basename(path)
    m = FOLDER_RE.search(name)
    if not m:
        return None
    ph, salt, suc = decode_ph(m.group(1)), int(m.group(2)), int(m.group(3))

    per_region = os.path.join(path, "pic_symmetry_regions", "per_region.csv")
    vs_r = os.path.join(path, "pic_SUC_exterior_vs_r.csv")
    for f in (per_region, vs_r):
        if not os.path.isfile(f):
            print(f"  skipping {name}: missing {os.path.relpath(f, path)}", file=sys.stderr)
            return None

    with open(per_region) as fh:
        regions = list(csv.DictReader(fh))
    reg3 = [float(r["mean"]) for r in regions if r["region_type"] == "3F"]
    reg5 = [float(r["mean"]) for r in regions if r["region_type"] == "5F"]

    # Whole-molecule Gamma_23 is the last row of the r-profile, i.e. the
    # value at the run's outermost r. NB: this is r_max, not a verified
    # plateau -- check pic_SUC_exterior_summary.png before trusting it.
    with open(vs_r) as fh:
        rows = list(csv.DictReader(fh))
    gamma = float(rows[-1]["gamma23_mean"])
    sem = float(rows[-1]["gamma23_sem"])
    rstar = float(rows[-1]["r_angstrom"])

    return dict(name=name, ph=ph, salt=salt, suc=suc, path=path,
                reg3=np.array(reg3), reg5=np.array(reg5),
                gamma=gamma, sem=sem, rstar=rstar)


def add_normalizer(cond):
    """Bulk mole ratio n3/n1 from the run's own metadata.

    Normalizing by the mole ratio (rather than the raw excipient count)
    also corrects for exterior volume differing between runs, since
    n1_exterior is proportional to that volume at fixed water density.
    """
    pkl = os.path.join(cond["path"], "pic_SUC_exterior.pkl")
    if not os.path.isfile(pkl):
        raise FileNotFoundError(f"{cond['name']}: --normalize needs {pkl}")
    with open(pkl, "rb") as fh:
        meta = pickle.load(fh)["metadata"]
    cond["mole_ratio"] = (meta["mean_exterior_excipient_per_frame"]
                          / meta["mean_exterior_water_per_frame"])
    return cond


def label_for(cond):
    # plain ASCII hyphen, not an em-dash: this string is both drawn into the
    # figure and printed to the console, and a cp1252 Windows console
    # mangles non-ASCII on the way out.
    tag = FORMULATIONS.get((cond["ph"], cond["salt"], cond["suc"]))
    suffix = f"  [{tag[0]} - rank {tag[1]}]" if tag else ""
    return f"pH{cond['ph']:g}  {cond['salt']}mM  {cond['suc']}%{suffix}"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=here,
                   help="directory holding the condition folders (default: this script's own directory)")
    p.add_argument("--out", default="symmetry_jitter.png", help="output PNG path")
    p.add_argument("--normalize", action="store_true",
                   help="divide by each run's bulk mole ratio to remove concentration scaling")
    p.add_argument("--corr", type=float, default=DEFAULT_CORR,
                   help=f"SEM inflation for frame correlation (default {DEFAULT_CORR}; use 1 to disable)")
    p.add_argument("--seed", type=int, default=7, help="jitter RNG seed")
    p.add_argument("--dpi", type=int, default=170)
    args = p.parse_args()

    subdirs = sorted(os.path.join(args.root, d) for d in os.listdir(args.root)
                     if os.path.isdir(os.path.join(args.root, d)))
    conds = [c for c in (load_condition(d) for d in subdirs) if c]
    if not conds:
        sys.exit(f"no condition folders with symmetry output found under {args.root}")
    print(f"loaded {len(conds)} condition(s) from {args.root}")

    xlabel_r = "per-residue PIC by region"
    xlabel_l = f"whole-capsid PIC"
    if args.normalize:
        # Both panels get the SAME normalizer -- divide out the bulk mole
        # ratio and nothing else. Each panel keeps the quantity it started
        # with (whole-capsid PIC on the left, per-residue PIC on the right),
        # just expressed per unit of bulk sucrose instead of as a raw count.
        for c in conds:
            add_normalizer(c)
            for k in ("reg3", "reg5", "gamma", "sem"):
                c[k] = c[k] / c["mole_ratio"]
        xlabel_r = "per-residue PIC by region  /  (n₃/n₁)"
        xlabel_l = "whole-capsid PIC  /  (n₃/n₁)"

    # formulations first (in stability-rank order), then DOE corners sorted
    # by score, with a blank row between the two blocks
    forms = sorted([c for c in conds if (c["ph"], c["salt"], c["suc"]) in FORMULATIONS],
                   key=lambda c: FORMULATIONS[(c["ph"], c["salt"], c["suc"])][1])
    does = sorted([c for c in conds if (c["ph"], c["salt"], c["suc"]) not in FORMULATIONS],
                  key=lambda c: -c["gamma"])

    plt.rcParams.update({
        "font.size": 12, "font.family": "Calibri",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
    })
    height = max(4.0, 0.52 * (len(conds) + 1) + 1.9)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, height),
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    rng = np.random.default_rng(args.seed)

    ordered = forms + [None] + does if (forms and does) else (forms or does)
    y, ticks, labels, pos = 0, [], [], {}
    for c in ordered:
        if c is None:
            y -= 1
            continue
        pos[id(c)] = y
        ticks.append(y)
        labels.append(label_for(c))
        y -= 1

    for ax, block, color in ((axL, forms, TEAL), (axL, does, AMBER),
                             (axR, forms, TEAL), (axR, does, AMBER)):
        if block:
            ys = [pos[id(c)] for c in block]
            ax.axhspan(max(ys) + 0.55, min(ys) - 0.55, color=color, alpha=0.05, zorder=0)

    for c in forms + does:
        yy = pos[id(c)]
        axL.errorbar(c["gamma"], yy, xerr=c["sem"] * args.corr, fmt="o", color=AMBER,
                     ecolor=AMBER, markersize=8, capsize=3.5, elinewidth=1.5, zorder=3)
        # 3F jittered above the row centre, 5F below, so the two groups stay
        # visually separable where they overlap
        for vals, col, sign in ((c["reg3"], TEAL, +1), (c["reg5"], PLUM, -1)):
            j = rng.uniform(0.06, 0.24, size=len(vals))
            axR.scatter(vals, yy + sign * j, s=13, color=col, alpha=0.4,
                        linewidths=0, zorder=2)
        axR.scatter([c["reg3"].mean()], [yy], s=44, color=TEAL,
                    edgecolors="white", linewidths=0.8, zorder=4)
        axR.scatter([c["reg5"].mean()], [yy], s=44, color=PLUM,
                    edgecolors="white", linewidths=0.8, zorder=4)

    axL.set_xlabel(xlabel_l)
    axL.set_title("Whole-capsid", fontsize=13, loc="left", pad=10)
    axR.set_xlabel(xlabel_r)
    axR.set_title("3-fold / 5-fold regions", fontsize=13, loc="left", pad=10)
    for ax, tl in ((axL, labels), (axR, [])):
        ax.axvline(0, color=GRID, linewidth=1, zorder=0)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tl, fontsize=10.5)
        ax.set_ylim(min(ticks) - 0.7, max(ticks) + 0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=AMBER,
               markersize=8, label="whole-capsid"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL,
               markersize=8, label="3-fold"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PLUM,
               markersize=8, label="5-fold"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=10.5,
        bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")

    # Per-condition 5F-3F gap vs its own uncertainty.
    #
    # SE is the two-sample standard error built from the scatter ACROSS
    # regions, treating each region as one independent observation. No
    # --corr here, by design -- see the module docstring.
    #
    # The concentration factor cancels from this test (the same divisor hits
    # both region groups and their errors), so the table is identical with
    # and without --normalize. That makes it the one conclusion in this
    # analysis that is immune to the whole normalization question.
    print("\n5F - 3F gap   (SE from between-region scatter; regions = unit of replication)")
    n_sig = 0
    for c in forms + does:
        gap = c["reg5"].mean() - c["reg3"].mean()
        se = np.sqrt(c["reg3"].var(ddof=1) / len(c["reg3"])
                     + c["reg5"].var(ddof=1) / len(c["reg5"]))
        ratio = abs(gap) / se if se > 0 else 0.0
        sig = ratio > 2
        n_sig += sig
        print(f"  {label_for(c):34s} gap={gap:+10.4g}  ±{se:8.4g}  "
              f"{ratio:5.2f}x  {'SIG' if sig else '-'}")
    print(f"\n  {n_sig}/{len(conds)} significant at 2xSE "
          f"(~{0.05*len(conds):.1f} expected by chance at alpha=0.05)")


if __name__ == "__main__":
    main()
