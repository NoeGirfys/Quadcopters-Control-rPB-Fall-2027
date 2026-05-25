#!/bin/bash
#SBATCH --job-name=baseline_offdir_cpu
#SBATCH --account=master          # Ton compte étudiant
#SBATCH --partition=academic      # Le couloir obligatoire pour les étudiants sur Jed
#SBATCH --qos=academic            # La règle de temps associée à cette partition
#SBATCH --cpus-per-task=4       # CPU-only -> on parallélise via les threads BLAS/torch
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/baseline_offdir_cpu_%j.out
#SBATCH --error=logs/baseline_offdir_cpu_%j.err

mkdir -p logs

module purge
module load gcc python
source venv_MAML_SCITAS/bin/activate
cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "CPUs/threads: ${SLURM_CPUS_PER_TASK:-16}"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| threads:', torch.get_num_threads())"

# --- Version CPU de run_baseline_izar_offdir.sh (run STANDALONE) ------
# Pas de --from-maml-ckpt : lançable EN MÊME TEMPS que le run MAML CPU.
# Comparaison strictement équitable car TOUS les paramètres de tirage des
# tâches sont identiques à run_maml_izar_offdir_cpu.sh :
#   * mêmes --tasks-config / --target-config (toute la distribution),
#   * mêmes --n-points-train/eval (10/10) et --k-samples,
#   * même --seed.
# => points (masses, positions, x0) IDENTIQUES AU BIT PRÈS + init identique.
# --n-inner-steps/--lr-inner/--inner-grad-clip rendent l'éval target par
# epoch SYMÉTRIQUE avec MAML.
#
# >>> IMPORTANT : si tu changes une valeur dans run_maml_izar_offdir_cpu.sh,
#     répercute-la EXACTEMENT ici (sinon les jeux de tâches divergent).
srun python -u training_MAML/train_baseline.py \
    --tasks-config  training_MAML/tasks_offdir.yaml \
    --target-config training_MAML/target_offdir.yaml \
    --dynamics nonlinear \
    --epochs 500 \
    --lr-outer 3e-4 \
    --hidden 64 \
    --n-points-train 10 \
    --n-points-eval  10 \
    --k-samples 5 \
    --n-inner-steps 1 \
    --lr-inner 0.05 \
    --inner-grad-clip 1.0 \
    --obs-noise-scale 2.0 \
    --tau-start 0.8 \
    --tau-end 2.0 \
    --t-sim 3.0 \
    --terminal-weight 50.0 \
    --pos-weight 10.0 \
    --z-weight 1.0 \
    --seed 42 \
    --tag izar_offdir_cpu
