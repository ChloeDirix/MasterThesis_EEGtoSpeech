#!/bin/bash
#SBATCH --job-name=SS_plots
#SBATCH --clusters=genius
#SBATCH --account=intro_vsc37381
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /user/leuven/373/vsc37381/data/anaconda3/bin/activate AADProjectEnv

CODE_DIR="$VSC_DATA/MasterThesis_EEGtoSpeech/AADProject"
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"
cd "$CODE_DIR"

ARRAY_JOB_ID="66203748"

RUN_DIR="$VSC_SCRATCH/Results_Lin/SS/run_${ARRAY_JOB_ID}"


python BackwardModel/RunBackwardModel_SS.py --merge-only --run-dir "$RUN_DIR"

# stage out the merged results + plots
rsync -avh "$RUN_DIR/" "$CODE_DIR/Results_Lin/SS/run_${ARRAY_JOB_ID}/"

echo "done."
