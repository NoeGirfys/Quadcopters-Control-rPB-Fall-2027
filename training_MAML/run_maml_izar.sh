#!/bin/bash
#SBATCH --job-name=maml_cf
#SBATCH --partition=gpu          # file d'attente GPU sur Izar
#SBATCH --qos=normal             # QOS normal (jusqu'à 3 jours sur Izar)
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --cpus-per-task=4        # CPU pour numpy / overhead Python
#SBATCH --mem=16G                # RAM
#SBATCH --time=04:00:00          # ~23s/epoch -> 445 epochs restantes en ~3h, marge incluse
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
srun python -u training_MAML/train_maml.py \
    --resume training_MAML/maml_linearized_h64_o1_n50_izar_uniform_tasks_inprogress.pt \
    --profile
