#!/bin/bash
#SBATCH --job-name=baseline_cf
#SBATCH --partition=gpu          # file d'attente GPU sur Izar
#SBATCH --qos=normal             # QOS normal (jusqu'à 3 jours sur Izar)
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --cpus-per-task=4        # CPU pour numpy / overhead Python
#SBATCH --mem=16G                # RAM
#SBATCH --time=02:00:00          # baseline batchée -> rapide, marge large
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# --- Créer le dossier de logs s'il n'existe pas encore ----------------
mkdir -p logs

# --- Environnement ----------------------------------------------------
module purge
module load gcc python

source venv_MAML_SCITAS/bin/activate

cd $SLURM_SUBMIT_DIR

# --- Vérification GPU -------------------------------------------------
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# --- Entraînement de la baseline --------------------------------------
# --from-maml-ckpt : charge les MÊMES tâches (positions de masses), les
# MÊMES starting points x0 et tous les hyperparamètres depuis le run MAML
# -> comparaison strictement équitable, indépendante du device CPU/GPU.
srun python -u training_MAML/train_baseline.py \
    --from-maml-ckpt training_MAML/maml_linearized_h64_o1_n50_izar_uniform_tasks_ep472.pt \
    --tag izar_uniform_tasks
