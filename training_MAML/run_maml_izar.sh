#!/bin/bash
#SBATCH --job-name=maml_cf
#SBATCH --partition=gpu          # file d'attente GPU sur Izar
#SBATCH --qos=normal             # QOS normal (jusqu'à 3 jours sur Izar)
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --cpus-per-task=4        # CPU pour numpy / overhead Python
#SBATCH --mem=16G                # RAM
#SBATCH --time=03:00:00          # N*M_train est plus petit qu'avant -> à mesurer en debug
#SBATCH --output=logs/maml_%j.out
#SBATCH --error=logs/maml_%j.err

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

# --- Lancement de l'entraînement (fresh run, jeu de tâches composite) -
#   --tasks-config  : YAML décrivant chaque tâche (sélection de gaussiennes
#                     de position de masse + octants de départ du drone)
#   --target-config : tâche held-out, sa loss est loggée par epoch mais
#                     n'influence jamais l'optimiseur
#   --mass-pos-sigma: écart-type isotrope des 4 gaussiennes autour des moteurs
#   --n-points-*    : nombre de points (support / query) tirés par tâche au
#                     démarrage, figés pour tout le run
srun python -u training_MAML/train_maml.py \
    --tasks-config  training_MAML/tasks_default.yaml \
    --target-config training_MAML/target_default.yaml \
    --dynamics nonlinear \
    --maml-order 1 \
    --epochs 500 \
    --lr-outer 3e-4 \
    --lr-inner 0.05 \
    --n-inner-steps 1 \
    --mass-min 0.002 \
    --mass-max 0.014 \
    --mass-pos-sigma 0.01 \
    --half-side 0.3 \
    --n-points-train 100 \
    --n-points-eval  50 \
    --obs-noise-scale 1.0 \
    --tau-div 1.0 \
    --t-sim 2.0 \
    --seed 42 \
    --tag izar_composite \
    --profile
