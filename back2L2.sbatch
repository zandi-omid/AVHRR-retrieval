#!/bin/bash
#SBATCH --account=behrangi
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=6G
#SBATCH --time=12:00:00
#SBATCH --job-name=L2_2018
#SBATCH --output=logs/L2_2018_%j.out
#SBATCH --error=logs/L2_2018_%j.err

set -euo pipefail

echo "========== JOB STARTED: $(date) =========="
echo "Node list: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "----------------------------------------------------------"

source /home/u20/omidzandi/micromamba/etc/profile.d/mamba.sh
micromamba activate retrieval

cd /xdisk/behrangi/omidzandi/AVHRR-retrieval

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

python AVHRR_collocation_pipeline/back2L2.py \
    --config config/back2L2_config.toml

echo "=========== JOB ENDED: $(date) ==========="