#!/bin/bash
#SBATCH --job-name=baseline_cf
#SBATCH --partition=gpu          # file d'attente GPU sur Izar
#SBATCH --qos=normal             # QOS normal (jusqu'à 3 jours sur Izar)
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --cpus-per-task=4        # CPU pour numpy / overhead Python
#SBATCH --mem=16G                # RAM
#SBATCH --time=01:40:00          # baseline batchée -> rapide, marge large
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

# --- Entraînement de la baseline --------------------------------------
# --from-maml-ckpt : charge les MÊMES task_set ET target_set (positions,
# masses, x0 — tous identiques au bit près) ainsi que tous les
# hyperparamètres depuis le run MAML correspondant. Comparaison
# strictement équitable, indépendante du device CPU/GPU.
#
# >>> Mets à jour le chemin du checkpoint après chaque run MAML.
srun python -u training_MAML/train_baseline.py \
    --from-maml-ckpt training_MAML/maml_nonlinear_h64_o1_n4_izar_no_mass_ep500.pt \
    --tag izar_no_mass
