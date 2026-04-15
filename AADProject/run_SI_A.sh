#!/bin/bash -l
#SBATCH --job-name=SI
#SBATCH --clusters=wice
#SBATCH --account=lp_edu_large_omics
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclude=s28c11n4
#SBATCH --exclude=s28c11n3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=1-34
#SBATCH --output=logs/SI_%A/%x_%A_%a.out
#SBATCH --error=logs/SI_%A/%x_%A_%a.err

set -euo pipefail
source /user/leuven/373/vsc37381/data/anaconda3/bin/activate AADProjectEnv

THREADS=8
export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS

mkdir -p logs/SI_$SLURM_ARRAY_JOB_ID

CODE_DIR="$VSC_DATA/MasterThesis_EEGtoSpeech/AADProject"
DATA_DIR="$VSC_SCRATCH"
export PROJECT_ROOT="$CODE_DIR"
export PROJECT_DATA_ROOT="$DATA_DIR"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"


RUN_DIR="$VSC_SCRATCH/Results_Lin/SI/run_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RUN_DIR"


cd "$CODE_DIR"

CONFIG_COPY="$RUN_DIR/config_used.yaml"
cp -n "$CODE_DIR/config.yaml" "$CONFIG_COPY"
export AAD_CONFIG="$CONFIG_COPY"

SUBJECT=$(python - <<'PY'
import yaml, os, sys
cfg_path = os.environ["AAD_CONFIG"]
cfg = yaml.safe_load(open(cfg_path))
use_subjects=cfg["use_subjects"]
subjects = cfg["subjects"][use_subjects]
idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
if idx < 0 or idx >= len(subjects):
    print(f"ERROR: array index out of range (1..{len(subjects)})", file=sys.stderr)
    sys.exit(1)
print(subjects[idx])
PY
)

echo "Using config: $AAD_CONFIG"
echo "Array task $SLURM_ARRAY_TASK_ID running subject: $SUBJECT"
echo "Run dir (scratch): $RUN_DIR"


srun --cpus-per-task=$THREADS --cpu-bind=cores python BackwardModel/RunBackwardModel_SI.py \
  --single-subject "$SUBJECT" \
  --run-dir "$RUN_DIR" 

echo "Done."