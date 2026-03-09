#!/bin/bash
#SBATCH --job-name=DLPlots
#SBATCH --clusters=genius
#SBATCH --account=intro_vsc37381
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logsDL/DLPlots_%A.out
#SBATCH --error=logsDL/DLPlots_%A.err


# ---- Activate env ----
source /user/leuven/373/vsc37381/data/anaconda3/bin/activate AADProjectEnv

# ---- Paths ----
CODE_DIR="$VSC_DATA/MasterThesis_EEGtoSpeech/AADProject"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

slurm_id="58999594"
slurm_id_lin="58959745"

RESULTS_DIR="/lustre1/scratch/373/vsc37381/Results_DL/run_$slurm_id"
lin_RESULTS="/lustre1/scratch/373/vsc37381/Results_Lin/SI/run_$slurm_id_lin/mTRF_summary_ALL.csv"

echo "Using CODE_DIR:    $CODE_DIR"
echo "Using RESULTS_DIR: $RESULTS_DIR"

python -u "$CODE_DIR/DLModel/plots_after.py" \
  --results-dir "$RESULTS_DIR" \
  --baseline-csv "$lin_RESULTS"

mkdir -p "$CODE_DIR/Results_DL"
rsync -av --delete "$RESULTS_DIR" "$CODE_DIR/Results_DL" 

