#!/bin/bash
#SBATCH --job-name=pqr_calc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --partition=mem-queue-multi-az
#SBATCH --nodelist=mem-queue-multi-az-dy-r8i-32xl-1
#SBATCH --exclusive

cd "${SLURM_SUBMIT_DIR}" || { echo "ERROR: cannot cd to SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"; exit 1; }
source /apps/gromacs/2025.3-gpu-mpi/bin/GMXRC.bash
source /apps/miniconda3/bin/activate apbs_env

mkdir -p logs

echo "=========================================================="
echo "PIC calculation (per-residue): ${EXCIPIENT}"
echo "Job: ${SLURM_JOB_ID}   Node: $(hostname)   Started: $(date)"
echo "=========================================================="



DIR="aav8ph6salt30"
echo "processing: ${DIR}"
printf "Protein\nSystem\n" | gmx_mpi trjconv -s aav8/${DIR}/md.tpr -f aav8/${DIR}/md.gro \
    -o aav8/${DIR}/step_cluster.gro -pbc cluster
printf "Protein\nSystem\n" | gmx_mpi trjconv -s aav8/${DIR}/md.tpr -f aav8/${DIR}/step_cluster.gro \
    -o aav8/${DIR}/md_whole.gro -pbc mol -center -ur compact

python build_suc_pqr.py --gro aav8/${DIR}/md_whole.gro --itp SUC_kb_reordered.itp \
    --protein-out /tmp/ignore.pdb --suc-out pqr/${DIR}/sucrose.pqr --suc-within 15

python build_pqr_parmed.py --top aav8/input/aav8_ph6/topol_${DIR}.top --gro aav8/${DIR}/md_whole.gro \
    --topdir ~/gmx-ff --protein-out pqr/${DIR}/capsid_protein.pqr --suc-out pqr/${DIR}/suc_parmed.pqr
cd ~/aav/pqr/${DIR}
cat capsid_protein.pqr sucrose.pqr > capsid_suc.pqr


EXIT_CODE=$?

echo "=========================================================="
echo "Finished: $(date)   Exit code: ${EXIT_CODE}"
echo "=========================================================="

exit ${EXIT_CODE}