#!/bin/bash
#SBATCH --job-name=AAD_preproc
#SBATCH --clusters=wice
#SBATCH --account=lp_edu_large_omics
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------- CONDA ----------------
source /user/leuven/373/vsc37381/data/anaconda3/bin/activate AADProjectEnv
export PYTHONPATH=/user/leuven/373/vsc37381/data/MasterThesis_EEGtoSpeech/AADProject

# ---------------- THREADING ----------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------------- PATHS ----------------
CODE_DIR="$VSC_DATA/MasterThesis_EEGtoSpeech/AADProject"

# Job-local scratch root
export PROJECT_ROOT=$VSC_SCRATCH
mkdir -p "$PROJECT_ROOT"

echo "Scratch project root: $PROJECT_ROOT"

# ---------------- STAGE-IN (DATA ONLY) ----------------
# copy config
rsync -avh "$CODE_DIR/config.yaml" "$PROJECT_ROOT/"


# create output folders on scratch
mkdir -p "$PROJECT_ROOT/Data_InputModelFine/EEG_PP"
mkdir -p "$PROJECT_ROOT/Data_InputModelFine/Envelopes"

# ---------------- RUN ----------------
cd "$CODE_DIR"

echo "Starting preprocessing..."
srun --cpu-bind=cores python DataPreparation.py

echo "Preprocessing finished."

# ---------------- STAGE-OUT ----------------
#echo "Copying NWB outputs back to DATA..."
#rsync -avh "$PROJECT_ROOT/Data_InputModel/" "$CODE_DIR/Data_InputModel/"

echo "Done."