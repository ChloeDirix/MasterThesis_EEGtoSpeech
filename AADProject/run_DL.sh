#!/bin/bash -l
#SBATCH --job-name=DL
#SBATCH --clusters=genius
#SBATCH --account=lp_eegsigpro
#SBATCH --nodes=1
#SBATCH --partition=gpu_p100
#SBATCH --gpus-per-node=1
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --array=15-18
#SBATCH --output=logsDL/DL_%A/%x_%A_%a.out
#SBATCH --error=logsDL/DL_%A/%x_%A_%a.err

set -euo pipefail

#sSBATCH --cpus-per-task=1


# ---------------- CONDA ----------------
source /user/leuven/373/vsc37381/data/anaconda3/bin/activate AADProjectEnv
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libLLVM-14.so${LD_PRELOAD:+:$LD_PRELOAD}"



# ---------------- THREADING ------------
THREADS=8
export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS

mkdir -p "logsDL/DL_${SLURM_ARRAY_JOB_ID}"


# ---------------- PATHS ----------------
CODE_DIR="$VSC_DATA/MasterThesis_EEGtoSpeech/AADProject"
DATA_DIR="$VSC_SCRATCH"
export PROJECT_ROOT="$CODE_DIR"
export PROJECT_DATA_ROOT="$DATA_DIR"
export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

 
RUN_dir="$VSC_SCRATCH/Results_DL/run_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RUN_dir"

cd "$CODE_DIR"

CONFIG_COPY="$RUN_dir/config_used.yaml"
[ -f "$CONFIG_COPY" ] || cp "$CODE_DIR/config.yaml" "$CONFIG_COPY"
export AAD_CONFIG="$CONFIG_COPY"


echo "============================================================"
echo "Starting training..."
echo "============================================================"


SUBJECT=$(python - <<'PY'
import yaml, os, sys
cfg = yaml.safe_load(open(os.environ["AAD_CONFIG"]))
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

print(subjects[idx], end="")  
PY
)

python -u DLModel/RunDLModel.py --test-subj "$SUBJECT" --results-dir "$RUN_dir"

echo "============================================================"
echo "Done."
echo "============================================================"