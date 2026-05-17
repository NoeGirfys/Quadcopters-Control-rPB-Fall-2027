#!/bin/bash
#SBATCH --job-name=maml_cf_dbg
#SBATCH --partition=gpu          # file d'attente GPU sur Izar
#SBATCH --qos=debug              # QOS debug : 1h max, haute priorité, gratuit
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --cpus-per-task=4        # CPU pour numpy / overhead Python
#SBATCH --mem=16G                # RAM
#SBATCH --time=00:20:00          # ~30-40 epochs : assez pour mesurer le temps/epoch
#SBATCH --output=logs/maml_dbg_%j.out
#SBATCH --error=logs/maml_dbg_%j.err

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

# --- Run debug : config IDENTIQUE au vrai run, mais tag distinct ------
# But : lire les lignes [profile ep N] pour mesurer le temps/epoch en
# nonlinear + masse variable, puis caler --time dans run_maml_izar.sh.
# Le job sera tué à 20 min ; les checkpoints "izar_dbg" sont jetables.
srun python -u training_MAML/train_maml.py \
    --dynamics nonlinear \
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
    --mass-min 0.002 \
    --mass-max 0.014 \
    --n-x0-train 128 \
    --n-x0-eval  64 \
    --obs-noise-scale 1.0 \
    --tau-div 1.0 \
    --t-sim 2.0 \
    --seed 42 \
    --tag izar_dbg \
    --profile
