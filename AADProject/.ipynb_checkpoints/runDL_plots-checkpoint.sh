#!/bin/bash -l
#SBATCH --job-name=DLPlots
#SBATCH --clusters=Wice
#SBATCH --account=lp_edu_large_omics
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclude=s28c11n4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logsDL/DLPlots_%A.out
#SBATCH --error=logsDL/DLPlots_%A.err

set -euo pipefail
shopt -s nullglob

# ---- Activate env ----
source /user/leuven/373/vsc37381/data/anaconda3/bin/activate AADProjectEnv

# ---- Paths ----
CODE_DIR="$VSC_DATA/MasterThesis_EEGtoSpeech/AADProject"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

slurm_id="59091633"
slurm_id_lin="66280388"

RESULTS_DIR="/lustre1/scratch/373/vsc37381/Results_DL/run_$slurm_id"
LIN_RESULTS="/lustre1/scratch/373/vsc37381/Results_Lin/SI/run_$slurm_id_lin/mTRF_summary_ALL.csv"
LOCAL_RUN="$CODE_DIR/Results_DL/run_$slurm_id"

echo "Using CODE_DIR:    $CODE_DIR"
echo "Using RESULTS_DIR: $RESULTS_DIR"
echo "Using LIN_RESULTS: $LIN_RESULTS"

mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/folds"
mkdir -p "$RESULTS_DIR/summary"
mkdir -p "$RESULTS_DIR/dl_vs_linear"


python -u "$CODE_DIR/DLModel/plots_after.py" \
  --results-dir "$RESULTS_DIR" \
  --out-dir "$RESULTS_DIR" \
  --baseline-csv "$LIN_RESULTS"

mkdir -p "$LOCAL_RUN"
rsync -av "$RESULTS_DIR/" "$LOCAL_RUN/"

echo "Done. Final structure exists in:"
echo "  scratch: $RESULTS_DIR"
echo "  data:    $LOCAL_RUN"