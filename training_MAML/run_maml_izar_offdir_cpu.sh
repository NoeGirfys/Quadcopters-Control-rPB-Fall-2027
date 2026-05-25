#!/bin/bash
#SBATCH --job-name=maml_offdir_cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4       # CPU-only -> on parallélise via les threads BLAS/torch
#SBATCH --mem=8G                 # 10/10 points -> empreinte mémoire faible
#SBATCH --time=03:00:00          # 10/10 points -> rapide ; marge large sur CPU
#SBATCH --output=logs/maml_offdir_cpu_%j.out
#SBATCH --error=logs/maml_offdir_cpu_%j.err

# --- Créer le dossier de logs s'il n'existe pas encore ----------------
mkdir -p logs

# --- Environnement ----------------------------------------------------
module purge
module load gcc python
source venv_MAML_SCITAS/bin/activate
# (PyYAML est requis pour lire les *.yaml ; au besoin : pip install pyyaml)

cd $SLURM_SUBMIT_DIR

# --- Threads CPU (torch/numpy n'ont pas de GPU ici) -------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "CPUs/threads: ${SLURM_CPUS_PER_TASK:-16}"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| threads:', torch.get_num_threads())"

# --- Version CPU de run_maml_izar_offdir.sh ---------------------------
#   Expérience "offset DIRECTION", MAIS avec --n-points-train 10 et
#   --n-points-eval 10 (au lieu de 100/50) pour tourner sur CPU.
#   Tag distinct (izar_offdir_cpu) -> n'écrase PAS les checkpoints du run GPU.
#   Toute la distribution de tâches est dans le YAML (source de vérité unique).
srun python -u training_MAML/train_maml.py \
    --tasks-config  training_MAML/tasks_offdir.yaml \
    --target-config training_MAML/target_offdir.yaml \
    --dynamics nonlinear \
    --maml-order 1 \
    --epochs 500 \
    --lr-outer 3e-4 \
    --lr-inner 0.05 \
    --n-inner-steps 1 \
    --n-points-train 10 \
    --n-points-eval  10 \
    --k-samples 5 \
    --obs-noise-scale 2.0 \
    --tau-start 0.8 \
    --tau-end 2.0 \
    --t-sim 3.0 \
    --seed 42 \
    --tag izar_offdir_cpu \
    --profile
