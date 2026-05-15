#!/bin/bash
#SBATCH --job-name=maml_cf
#SBATCH --partition=gpu          # file d'attente GPU sur Izar
#SBATCH --qos=normal             # Priorité et limites (recommandé par SCITAS)
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --cpus-per-task=4        # CPU pour le dataloader / numpy / PyBullet
#SBATCH --mem=16G                # RAM
#SBATCH --time=08:00:00          # Temps max autorisé
#SBATCH --output=logs/maml_%j.out
#SBATCH --error=logs/maml_%j.err

# --- Créer le dossier de logs s'il n'existe pas encore ----------------
mkdir -p logs

# --- Environnement ----------------------------------------------------
module purge
module load gcc python

# On active l'environnement créé tout à l'heure
source venv_MAML_SCITAS/bin/activate

# Aller dans le bon répertoire (optionnel mais très sûr)
cd $SLURM_SUBMIT_DIR

# --- Vérification GPU -------------------------------------------------
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# --- Lancement de l'entraînement --------------------------------------
srun python training_MAML/train_maml.py \
    --dynamics linearized \
    --maml-order 1 \
    --epochs 500 \
    --lr-outer 3e-4 \
    --lr-inner 0.05 \
    --n-inner-steps 1 \
    --n-tasks 50 \
    --xy-min -0.04 \
    --xy-max  0.04 \
    --z-min  -0.01 \
    --z-max   0.01 \
    --mass 0.010 \
    --n-x0-train 128 \
    --n-x0-eval  64 \
    --obs-noise-scale 1.0 \
    --tau-div 1.0 \
    --t-sim 2.0 \
    --seed 42 \
    --tag izar_uniform_tasks
