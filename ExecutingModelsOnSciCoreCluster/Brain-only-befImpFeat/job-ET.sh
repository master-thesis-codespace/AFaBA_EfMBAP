#!/bin/bash
# =============================================================================
# Example SLURM batch script for a single-GPU PyTorch job.
#
# WHAT IS THIS FILE?
#   This is a "job script". You do NOT run it directly like a normal script.
#   Instead you SUBMIT it to the cluster's scheduler (SLURM) with:
#
#       sbatch job_test.sh
#
#   SLURM then finds a free machine (a "compute node") that matches the
#   resources you ask for below, runs the script there, and saves the output
#   to files. See SLURM_CHEATSHEET.md in this folder for the commands you
#   need day to day (submit, check, cancel, etc.).
#
# HOW THE FILE IS STRUCTURED:
#   1. The "#SBATCH" lines at the top tell SLURM what resources you need.
#      These look like comments to bash (they start with #), but SLURM reads
#      them. They MUST come before any real command. Once a normal command
#      runs, SLURM stops reading #SBATCH lines.
#   2. Everything after that is a normal bash script that runs on the
#      compute node once your job starts.
# =============================================================================

# ---------------------------------------------------------------------------
# SLURM resource requests
# ---------------------------------------------------------------------------
#SBATCH --job-name=BOTH_ET_beImpFeat_GAF-GS9         # A name for your job. Shows up in `squeue`
                                        #   and is used in the log filenames below (%x).
#SBATCH --output=logs/%x_%j.out         # Where normal output (stdout) is written.
                                        #   %x = job name, %j = job ID number.
                                        #   e.g. logs/pytom_123456.out
#SBATCH --error=logs/%x_%j.err          # Where error output (stderr) is written.
                                        #   Tip: you can combine both into one file by
                                        #   only setting --output and deleting --error.
#SBATCH --partition=scicore             # Which group of machines to run on.
                                        #   Use either rtx4090 or a100 (rtx4090 is usually enough).
                                        #   A "partition" = a queue tied to a set of nodes.
                                        #   See available ones with: sinfo
#SBATCH --gres=gpu:0                    # Generic RESource. Here: request 1 GPU.
                                        #   "gpu:2" would request 2 GPUs.
                                        #   WITHOUT this line you get NO GPU, even on a GPU node.
#SBATCH --cpus-per-task=32               # How many CPU cores to give the job (for data
                                        #   loading, pre-processing, etc.). 8 is a sensible default.
#SBATCH --qos=1week                      # Quality Of Service = the time/priority class.
                                        #   One of: gpu30min, gpu6hours, gpu1day, gpu1week.
                                        #   Pick the SHORTEST one that fits your runtime: shorter
                                        #   QOS usually means your job starts sooner. The QOS must
                                        #   be consistent with --time below.
#SBATCH --mem=32GB                     # Total RAM (CPU memory, NOT GPU memory) for the job.
                                        #   If your job uses more than this, SLURM kills it.
                                        #   256GB is generous; lower it if you don't need that much.
#SBATCH --time=60:00:00                  # Maximum wall-clock time (HH:MM:SS). When this is hit,
                                        #   SLURM kills the job even if it's not finished. Must be
                                        #   <= the limit of the chosen --qos (here: 6 hours).
#SBATCH --nodes=1                       # Number of machines. 1 is correct for a single-GPU job.
#SBATCH --ntasks-per-node=1             # Number of parallel "tasks" (processes) per node.
                                        #   1 task = one python process here. (Relevant for MPI /
                                        #   multi-process jobs; leave at 1 for a normal script.)

# ---------------------------------------------------------------------------
# Bash safety settings (these run on the compute node, not on SLURM)
# ---------------------------------------------------------------------------
# -e : stop the whole script immediately if any command fails (non-zero exit).
# -u : treat use of an undefined variable as an error (catches typos).
# -o pipefail : if any command in a pipe (a | b) fails, the whole pipe fails.
# Together these make the job fail loudly and early instead of silently
# continuing in a broken state.
set -euo pipefail

############################################
# User settings  -- edit these for your setup
############################################
CONDA_BASE="$HOME/miniconda3"            # Path to your Anaconda/Miniconda install.
                                        #   $HOME is your home directory. If you install
                                        #   miniconda with the provided setup_miniconda.sh,
                                        #   this would be "$HOME/miniconda3" instead.
CONDA_ENV="MRI"                        # Name of the conda environment to activate.
                                        #   This env must already contain the packages you
                                        #   need (e.g. pytorch). Create it once on the login
                                        #   node before submitting (see setup_miniconda.sh).

# Make sure the log directory exists, otherwise SLURM can't write the
# --output / --error files above and the job fails before it even starts.
mkdir -p logs

# ---------------------------------------------------------------------------
# Activate the conda environment
# ---------------------------------------------------------------------------
# `conda activate` only works after this `source` line in a non-interactive
# script like this one. Without it you'll get "conda: command not found" or
# "Your shell has not been properly configured to use conda activate".
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"



# ---------------------------------------------------------------------------
# Preflight CUDA check
# ---------------------------------------------------------------------------
# A quick sanity check that runs a small inline python snippet to confirm the
# GPU is actually visible to PyTorch BEFORE you launch a long computation.
# If "cuda available" prints False, stop and fix your environment / request
# rather than wasting hours of GPU time.
#
# `srun` launches the command as a SLURM "step" on the allocated node. Inside
# an sbatch job it's optional for a single command, but it's good practice and
# required if you ever scale to multiple tasks/nodes.
#
# The <<'PY' ... PY is a "heredoc": everything between the markers is fed to
# python's stdin as a script. The quotes around 'PY' stop bash from expanding
# variables inside it.

#srun python - <<'PY'
#import os
#import torch

#print("torch", torch.__version__)
#print("cuda available", torch.cuda.is_available())
#print("device count", torch.cuda.device_count())
#print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
#PY

# ---------------------------------------------------------------------------
# Your actual work goes here.
# ---------------------------------------------------------------------------
# Replace the line below with the real command you want to run, e.g.:
#   srun python train.py --epochs 100 --data /path/to/data
TARGET="BOTH_ET_beImpFeat_GAF-GS9"
srun python ExtraTree_age_model_modelling-GAF.py \
  --input BOTH_befImpFeat-ET.csv \
  --output_dir "$TARGET" \
  --n_estimators 1962 1965 1967 \
  --max_depth 36 37 38 \
  --min_samples_split 2 3 \
  --max_features 1.0 \
  --bootstrap "False" \
  --min_samples_leaf 1 2 3 \
  --criterion "squared_error"
echo "Deleting the model to free storage ..."
rm -rfv "$TARGET/best/models"
echo "Models successfully deleted."
echo "Moving log files to output directory ..."
mv logs/"${TARGET}"*.out "${TARGET}"/
mv logs/"${TARGET}"*.err "${TARGET}"/