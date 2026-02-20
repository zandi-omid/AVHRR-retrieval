#!/bin/bash
set -euo pipefail

echo "Submitting SLURM pipeline at: $(date)"

jid1=$(sbatch run_retrieval_parallel.sbatch | awk '{print $4}')
echo "Submitted run_retrieval_parallel.sbatch → JobID: $jid1"

jid2=$(sbatch --dependency=afterok:$jid1 /xdisk/behrangi/omidzandi/retrieved_maps/compress_job.sbatch | awk '{print $4}')
echo "Submitted compress_job.sbatch → JobID: $jid2"

jid3=$(sbatch --dependency=afterok:$jid2 /xdisk/behrangi/omidzandi/retrieved_maps/export_to_Rain.sbatch | awk '{print $4}')
echo "Submitted export_to_Rain.sbatch → JobID: $jid3"

jid4=$(sbatch --dependency=afterok:$jid3 /xdisk/behrangi/omidzandi/retrieved_maps/untar_on_rain.sbatch | awk '{print $4}')
echo "Submitted untar_on_rain.sbatch → JobID: $jid4"

echo "Pipeline submission complete."
echo "retrieve → compress → export → untar"