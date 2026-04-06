#!/bin/bash
#SBATCH --account=xdong
#SBATCH --partition=gpu_standard
#SBATCH --nodes=5
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --job-name=2010
#SBATCH --output=logs/2010_%j.out
#SBATCH --error=logs/2010_%j.err
#SBATCH --exclusive

export SLURM_NO_AUDIT=1
export SLURM_DISABLE_FATTR=1

echo "========== JOB STARTED: $(date) =========="
echo "Node list: $SLURM_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo "----------------------------------------------------------"

source /home/u20/omidzandi/micromamba/etc/profile.d/mamba.sh
micromamba activate retrieval

cd /xdisk/behrangi/omidzandi/AVHRR-retrieval

# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export REPROJ_THREADS=1
export MAX_PENDING_REPROJ=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_TIMEOUT=1800

export TORCH_NUM_THREADS=1

nvidia-smi || true
echo "----------------------------------------------------------"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

srun --export=ALL python AVHRR_collocation_pipeline/run_retrieval_parallel.py \
    --config config/retrieve_config.toml

echo "=========== JOB ENDED: $(date) ==========="