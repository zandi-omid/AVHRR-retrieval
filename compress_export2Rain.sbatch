#!/bin/bash
set -euo pipefail

echo "Submitting post-processing pipeline at: $(date)"

jid1=$(sbatch /xdisk/behrangi/omidzandi/retrieved_maps/compress_job.sbatch | awk '{print $4}')
echo "Submitted compress_job.sbatch → JobID: $jid1"

jid2=$(sbatch --dependency=afterok:$jid1 /xdisk/behrangi/omidzandi/retrieved_maps/export_to_Rain.sbatch | awk '{print $4}')
echo "Submitted export_to_Rain.sbatch → JobID: $jid2"

echo "Post-processing pipeline submission complete."
echo "compress → export"