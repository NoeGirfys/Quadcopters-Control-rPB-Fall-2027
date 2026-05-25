#!/bin/bash
#SBATCH --job-name=maml_offdir
#SBATCH --partition=gpu          # file d'attente GPU sur Izar
#SBATCH --qos=normal             # QOS normal (jusqu'à 3 jours sur Izar)
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --cpus-per-task=4        # CPU pour numpy / overhead Python
#SBATCH --mem=16G                # RAM
#SBATCH --time=05:30:00          # ~ comme le run payload (3 tâches, t_sim 3s)
#SBATCH --output=logs/maml_offdir_%j.out
#SBATCH --error=logs/maml_offdir_%j.err

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

# --- Lancement de l'entraînement : expérience "offset DIRECTION" ------
#   Axe par tâche = la DIRECTION d'une masse décalée de 6 g (M1, M2, M3),
#   target held-out = direction M4. Contrairement à la masse centrée,
#   l'adaptation est ici UTILE (la direction de l'offset est une info que le
#   feedforward exploite et qu'une policy fixe n'a pas), donc le méta-init
#   de MAML devrait battre la baseline (adaptée ET non adaptée).
#   6 g = sous la falaise de saturation du cap moteur (~8-9 g sur le bras).
#   Toute la distribution (magnitude, sigma, half_side) est dans le YAML.
#
#   Évaluation après coup :
#     python eval_gap.py --maml-ckpt maml_..._izar_offdir_ep500.pt \
#         --baseline-ckpt baseline_..._izar_offdir_ep500.pt --mode offset
srun python -u training_MAML/train_maml.py \
    --tasks-config  training_MAML/tasks_offdir.yaml \
    --target-config training_MAML/target_offdir.yaml \
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
    --tag izar_offdir \
    --profile
