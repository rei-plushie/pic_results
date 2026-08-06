#!/bin/bash
#SBATCH --job-name=sasa_calc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --partition=gen-queue-multi-az
#SBATCH --nodelist=gen-queue-multi-az-dy-m8i-32xl-2
#SBATCH --exclusive

cd "${SLURM_SUBMIT_DIR}" || { echo "ERROR: cannot cd to SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"; exit 1; }
source /apps/gromacs/2025.3-gpu-mpi/bin/GMXRC.bash

mkdir -p logs

echo "=========================================================="
echo "PIC calculation (per-residue): ${EXCIPIENT}"
echo "Job: ${SLURM_JOB_ID}   Node: $(hostname)   Started: $(date)"
echo "=========================================================="

DIRS=("10ns_aav1ph6salt30suc1" "10ns_aav1ph6salt30suc10")

for DIR in "${DIRS[@]}"; do
    if [ -d aav1/"$DIR" ]; then
        echo "processing: ${DIR}, $(date)"
        gmx_mpi sasa -s aav1/${DIR}/md.tpr -f aav1/${DIR}/md_whole.xtc -surface 'group "Protein" or resname SUC' \
            -output 'group "Protein"' -o sasa/${DIR}/capsid_sasa.xvg -or sasa/${DIR}/capsid_sasa_perres.xvg -probe 0.14 -ndots 24 -dt 100
        echo "done: $(date)"
    else
        echo "warning: ${DIR} not found"
    fi
done