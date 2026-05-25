#!/bin/bash
#SBATCH --job-name=baseline_offdir
#SBATCH --partition=gpu          # file d'attente GPU sur Izar
#SBATCH --qos=normal             # QOS normal (jusqu'à 3 jours sur Izar)
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --cpus-per-task=4        # CPU pour numpy / overhead Python
#SBATCH --mem=16G                # RAM
#SBATCH --time=04:00:00          # baseline batchée -> rapide, marge large
#SBATCH --output=logs/baseline_offdir_%j.out
#SBATCH --error=logs/baseline_offdir_%j.err

mkdir -p logs

module purge
module load gcc python
source venv_MAML_SCITAS/bin/activate
cd $SLURM_SUBMIT_DIR

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# --- Baseline "offset DIRECTION" (run STANDALONE, lançable en parallèle) -
# Pas de --from-maml-ckpt : la baseline et le run MAML peuvent tourner EN
# MÊME TEMPS. Comparaison strictement équitable car TOUS les paramètres de
# tirage des tâches sont identiques à run_maml_izar_offdir.sh :
#   * mêmes --tasks-config / --target-config (toute la distribution est dans
#     le YAML : direction par tâche, magnitude, mass_pos_sigma, half_side),
#   * mêmes --n-points-train/eval et --k-samples,
#   * même --seed.
# => points (masses, positions, x0) IDENTIQUES AU BIT PRÈS + init policy
# identique. --n-inner-steps/--lr-inner/--inner-grad-clip rendent l'éval
# target par epoch SYMÉTRIQUE avec MAML (adaptation K-shot puis query).
#
# >>> IMPORTANT : si tu changes une valeur dans run_maml_izar_offdir.sh,
#     répercute-la EXACTEMENT ici (sinon les jeux de tâches divergent).
srun python -u training_MAML/train_baseline.py \
    --tasks-config  training_MAML/tasks_offdir.yaml \
    --target-config training_MAML/target_offdir.yaml \
    --dynamics nonlinear \
    --epochs 500 \
    --lr-outer 3e-4 \
    --hidden 64 \
    --n-points-train 100 \
    --n-points-eval  50 \
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
    --tag izar_offdir
