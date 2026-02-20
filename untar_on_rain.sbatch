#!/bin/bash
#SBATCH --job-name=untar_rain
#SBATCH --output=logs/untar_rain_%j.out
#SBATCH --error=logs/untar_rain_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=standard
#SBATCH --account=behrangi

set -euo pipefail

REMOTE_HOST="omidzandi@rain.has.arizona.edu"
REMOTE_BASE="/ra1/pubdat/AVHRR_CloudSat_proj/SIMON_CODES/retrieved_archive"

ARCHIVE_NAME="2010.tar.gz"
DEST_DIR="/scratch/omidzandi/AVHRR_retrieved_archive/2010"
LOG_OUT="$REMOTE_BASE/logs/untar_2010.out"

ssh -o BatchMode=yes "$REMOTE_HOST" "
  set -euo pipefail
  cd '$REMOTE_BASE'
  nohup ./untar.sh '$REMOTE_BASE/$ARCHIVE_NAME' '$DEST_DIR' > '$LOG_OUT' 2>&1 &
  echo 'Started untar on Rain (nohup background). Log: $LOG_OUT'
"