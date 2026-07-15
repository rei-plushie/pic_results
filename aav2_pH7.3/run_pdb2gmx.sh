#!/bin/sh
# Generated for --merge no; do not run this answers.txt against a
# differently-merged pdb2gmx invocation -- the prompt order won't match.
"/c/Program Files/gmx/bin/gmx" pdb2gmx -f aav2_relabeled.pdb -o aav2_pH7.3/processed.gro -p aav2_pH7.3/topol.top -i aav2_pH7.3/posre.itp -ff charmm36-feb2026_cgenff-5.0 -water tip3p -lys -asp -glu -his -ter < aav2_pH7.3/answers.txt
