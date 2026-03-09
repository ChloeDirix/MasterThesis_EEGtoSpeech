#!/bin/bash
#SBATCH --job-name=SS
#SBATCH --clusters=genius
#SBATCH --account=intro_vsc37381
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-34
#SBATCH --output=logs/SS_%A/%x_%A_%a.out
#SBATCH --error=logs/SS_%A/%x_%A_%a.err

# ---------------- CONDA ----------------
source /user/leuven/373/vsc37381/data/anaconda3/bin/activate AADProjectEnv

win_len=20

# ---------------- THREADING ------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- logs dir ----
mkdir -p logs/SS_$SLURM_ARRAY_JOB_ID


# ---------------- PATHS ---------------
CODE_DIR="$VSC_DATA/MasterThesis_EEGtoSpeech/AADProject"
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"


# ---------------- SCRATCH --------------
export PROJECT_ROOT="$VSC_SCRATCH"


RUN_DIR="$VSC_SCRATCH/Results_Lin/SS/run_${SLURM_ARRAY_JOB_ID}"

mkdir -p "$RUN_DIR"


cd "$CODE_DIR"

cp -f "$CODE_DIR/config.yaml" "$RUN_DIR/config.yaml"
export AAD_CONFIG="$RUN_DIR/config.yaml"


# ---------------- PICK SUBJECT ----------
SUBJECT=$(python - <<'PY'
import yaml, os
cfg = yaml.safe_load(open("config.yaml"))
subjects = cfg["subjects"]["all"]
idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
print(subjects[idx])
PY
)

echo "Array task $SLURM_ARRAY_TASK_ID running subject: $SUBJECT"
echo "Run dir: $RUN_DIR"

# ---------------- RUN -------------------
srun --cpu-bind=cores python BackwardModel/RunBackwardModel_SS.py \
  --single-subject "$SUBJECT" \
  --run-dir "$RUN_DIR" \
  --window-s "$win_len"


echo "Done."
