#!/bin/bash
#SBATCH --job-name=baseline_cf
#SBATCH --partition=gpu          # file d'attente GPU sur Izar
#SBATCH --qos=normal             # QOS normal (jusqu'à 3 jours sur Izar)
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --cpus-per-task=4        # CPU pour numpy / overhead Python
#SBATCH --mem=16G                # RAM
#SBATCH --time=04:30:00          # baseline batchée -> rapide, marge large
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

mkdir -p logs

module purge
module load gcc python
source venv_MAML_SCITAS/bin/activate
cd $SLURM_SUBMIT_DIR

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# --- Entraînement de la baseline (run STANDALONE, lançable en parallèle) -
# On n'utilise PAS --from-maml-ckpt ici : la baseline et le run MAML
# peuvent donc tourner EN MÊME TEMPS. La comparaison reste strictement
# équitable car TOUS les paramètres de tirage des tâches sont identiques
# à run_maml_izar.sh :
#   * mêmes --tasks-config / --target-config,
#   * mêmes --mass-min/max, --mass-pos-sigma, --half-side,
#   * mêmes --n-points-train/eval et --k-samples,
#   * même --seed.
# Comme le sampler n'utilise que np.random.default_rng(seed) (tâches) et
# default_rng(seed+1) (target), des graines + configs + tailles identiques
# produisent des points (masses, positions, x0) IDENTIQUES AU BIT PRÈS,
# et l'init de la policy (torch.manual_seed(seed) + biais hover sur la masse
# moyenne) est elle aussi identique.
#
# --n-inner-steps / --lr-inner / --inner-grad-clip : rendent l'éval target
# par epoch SYMÉTRIQUE avec MAML (la baseline est adaptée K-shot sur la
# target puis évaluée sur la query, adaptation oubliée entre epochs), donc
# les deux courbes target_loss sont directement comparables.
#
# >>> IMPORTANT : si tu changes une valeur dans run_maml_izar.sh, répercute
#     la EXACTEMENT ici (sinon les jeux de tâches divergent).
srun python -u training_MAML/train_baseline.py \
    --tasks-config  training_MAML/tasks_default.yaml \
    --target-config training_MAML/target_default.yaml \
    --dynamics nonlinear \
    --epochs 500 \
    --lr-outer 3e-4 \
    --hidden 64 \
    --mass-min 0.010 \
    --mass-max 0.010 \
    --mass-pos-sigma 0.003 \
    --half-side 0.5 \
    --n-points-train 100 \
    --n-points-eval  50 \
    --k-samples 5 \
    --n-inner-steps 1 \
    --lr-inner 0.05 \
    --inner-grad-clip 1.0 \
    --obs-noise-scale 2.0 \
    --tau-start 0.8 \
    --tau-end 2.0 \
    --t-sim 2.0 \
    --terminal-weight 50.0 \
    --pos-weight 10.0 \
    --z-weight 1.0 \
    --seed 42 \
    --tag izar_offset10g
