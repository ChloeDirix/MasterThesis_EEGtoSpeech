#!/bin/bash
#SBATCH --job-name=DL
#SBATCH --clusters=genius
#SBATCH --account=intro_vsc37381
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=1-16
#SBATCH --output=logsDL/DL_%A/%x_%A_%a.out
#SBATCH --error=logsDL/DL_%A/%x_%A_%a.err

set -euo pipefail



# ---------------- CONDA ----------------
source /user/leuven/373/vsc37381/data/anaconda3/bin/activate AADProjectEnv
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libLLVM-14.so${LD_PRELOAD:+:$LD_PRELOAD}"



# ---------------- THREADING ------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p "logsDL/DL_${SLURM_ARRAY_JOB_ID}"


# ---------------- PATHS ----------------
CODE_DIR="$VSC_DATA/MasterThesis_EEGtoSpeech/AADProject"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

 
# ---------------- SCRATCH -------------
export PROJECT_ROOT="$VSC_SCRATCH"

RUN_dir="$VSC_SCRATCH/Results_DL/run_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RUN_dir"

cd "$CODE_DIR"

cp -f "$CODE_DIR/config.yaml" "$RUN_dir/config.yaml"
export AAD_CONFIG="$RUN_dir/config.yaml"


echo "============================================================"
echo "Starting training..."
echo "============================================================"


SUBJECT=$(python - <<'PY'
import yaml, os, sys
cfg = yaml.safe_load(open("config.yaml"))
subjects = cfg.get("subjects", None)
if subjects is None:
    print("ERROR: config has no 'subjects' key", file=sys.stderr)
    sys.exit(1)

if isinstance(subjects, dict):
    subjects = subjects.get("all", next((v for v in subjects.values() if isinstance(v, list)), None))

if not isinstance(subjects, list):
    print(f"ERROR: subjects is {type(subjects)} not a list", file=sys.stderr)
    sys.exit(1)

idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
if idx < 0 or idx >= len(subjects):
    print(f"ERROR: array index out of range (1..{len(subjects)})", file=sys.stderr)
    sys.exit(1)

print(subjects[idx], end="")  # IMPORTANT: no extra newline(s)
PY
)

srun python -u DLModel/RunDLModel.py --test-subj "$SUBJECT" --results-dir "$RUN_dir"


echo "============================================================"
echo "Done."
echo "Log: $RUN_dir/train.log"
echo "============================================================"