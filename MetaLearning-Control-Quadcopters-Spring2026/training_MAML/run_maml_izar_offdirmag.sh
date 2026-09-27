#!/bin/bash
#SBATCH --job-name=maml_offdirmag
#SBATCH --partition=gpu          # file d'attente GPU sur Izar
#SBATCH --qos=normal             # QOS normal (jusqu'à 3 jours sur Izar)
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --cpus-per-task=4        # CPU pour numpy / overhead Python
#SBATCH --mem=16G                # RAM
#SBATCH --time=06:00:00          # 6 tâches (~2x offdir) -> mesure via le run debug
#SBATCH --output=logs/maml_offdirmag_%j.out
#SBATCH --error=logs/maml_offdirmag_%j.err

# --- Créer le dossier de logs s'il n'existe pas encore ----------------
mkdir -p logs

# --- Environnement ----------------------------------------------------
module purge
module load gcc python
source venv_MAML_SCITAS/bin/activate
# (PyYAML est requis pour lire les *.yaml ; au besoin : pip install pyyaml)

cd $SLURM_SUBMIT_DIR

# --- Vérification GPU -------------------------------------------------
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# --- Lancement : expérience "offset DIRECTION x MAGNITUDE" (variant b) -
#   6 tâches = {M1,M2,M3} x {3 g, 6 g}, target held-out = M4 @ 5 g (OOD sur
#   direction ET magnitude, interpolable). Croise les deux axes pour pénaliser
#   au maximum la baseline-compromis : direction (feedforward transitoire) +
#   magnitude (poussée collective). half_side 0.2 -> manœuvre + tenue.
#   Toute la distribution de tâches est dans le YAML (source de vérité unique).
#
#   Évaluation après coup :
#     python eval_gap.py --maml-ckpt maml_..._izar_offdirmag_ep500.pt \
#         --baseline-ckpt baseline_..._izar_offdirmag_ep500.pt --mode offset
srun python -u training_MAML/train_maml.py \
    --tasks-config  training_MAML/tasks_offdirmag.yaml \
    --target-config training_MAML/target_offdirmag.yaml \
    --dynamics nonlinear \
    --maml-order 1 \
    --epochs 500 \
    --lr-outer 3e-4 \
    --lr-inner 0.05 \
    --n-inner-steps 1 \
    --n-points-train 100 \
    --n-points-eval  50 \
    --k-samples 5 \
    --obs-noise-scale 2.0 \
    --tau-start 0.8 \
    --tau-end 2.0 \
    --t-sim 3.0 \
    --seed 42 \
    --tag izar_offdirmag \
    --profile
