#!/bin/bash
#SBATCH --job-name=surfrad_calc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --partition=gen-queue-multi-az
#SBATCH --nodelist=gen-queue-multi-az-dy-m8i-32xl-1
#SBATCH --exclusive

cd "${SLURM_SUBMIT_DIR}" || { echo "ERROR: cannot cd to SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"; exit 1; }
source /apps/gromacs/2025.3-gpu-mpi/bin/GMXRC.bash
source /apps/miniconda3/bin/activate apbs_env


mkdir -p logs

echo "=========================================================="
echo "PIC calculation (per-residue): ${EXCIPIENT}"
echo "Job: ${SLURM_JOB_ID}   Node: $(hostname)   Started: $(date)"
echo "=========================================================="

DIRS=("10ns_aav1ph6salt200suc1" "10ns_aav1ph8salt200suc1" "10ns_aav1ph6salt200suc10" "10ns_aav1ph8salt200suc10")

for DIR in "${DIRS[@]}"; do
    if [ -d aav1/"$DIR" ]; thensba
        echo "processing: ${DIR}, $(date)"
        python3 surface_radial_density.py --top aav1/${DIR}/md.gro --xtc aav1/${DIR}/md_whole.xtc --water-atom OW \
            --excipient SUC --max-shell 15 --stride 20 --out surfrad/${DIR}/surface_radial_density.csv
        echo "done: $(date)"
    else
        echo "warning: ${DIR} not found"
    fi
done