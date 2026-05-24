#!/bin/bash
#SBATCH --job-name=maml_cf
#SBATCH --partition=gpu          # file d'attente GPU sur Izar
#SBATCH --qos=normal             # QOS normal (jusqu'à 3 jours sur Izar)
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --cpus-per-task=4        # CPU pour numpy / overhead Python
#SBATCH --mem=16G                # RAM
#SBATCH --time=04:30:00          # N*M_train est plus petit qu'avant -> à mesurer en debug
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
#   Expérience "payload magnitude" : l'axe par tâche est la MAGNITUDE d'une
#   masse CENTRÉE (~3 / 8 / 13 g), target held-out = 11 g. Le thrust collectif
#   (que le NN fixe, sans intégrateur car il remplace la boucle Vel-Z) doit
#   différer par tâche -> une policy unique ne peut pas tenir tous les poids
#   sans erreur statique en z ; MAML adapte le biais de poussée par tâche.
#   TOUTE la distribution de tâches (pool de magnitudes, sigma, half_side) est
#   dans le YAML -> source de vérité unique, le baseline pointe le même fichier.
#
#   --tasks-config  : YAML (3 axes : position, octants, magnitude par tâche)
#   --target-config : tâche held-out, sa loss (après adaptation K-shot) est
#                     loggée par epoch mais n'influence jamais l'optimiseur
#   --n-points-*    : nombre de points (support / query) tirés par tâche
#   --k-samples     : budget few-shot d'adaptation sur la target
#   --t-sim 3.0     : maintien long -> le coût est dominé par la phase de tenue
#                     (là où la différence de poussée par tâche se voit)
srun python -u training_MAML/train_maml.py \
    --tasks-config  training_MAML/tasks_default.yaml \
    --target-config training_MAML/target_default.yaml \
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
    --tag izar_payload \
    --profile
