#!/bin/bash
#SBATCH --job-name=maml_cf_dbg
#SBATCH --partition=gpu          # file d'attente GPU sur Izar
#SBATCH --qos=debug              # QOS debug : 1h max, haute priorité, gratuit
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --cpus-per-task=4        # CPU pour numpy / overhead Python
#SBATCH --mem=16G                # RAM
#SBATCH --time=00:10:00          # assez pour mesurer le temps/epoch
#SBATCH --output=logs/maml_dbg_%j.out
#SBATCH --error=logs/maml_dbg_%j.err

mkdir -p logs

module purge
module load gcc python
source venv_MAML_SCITAS/bin/activate
cd $SLURM_SUBMIT_DIR

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# --- Run debug : config IDENTIQUE au vrai run, mais tag distinct ------
# But : lire les lignes [profile ep N] pour mesurer le temps/epoch puis
# caler --time dans run_maml_izar.sh. Le job sera tué à 20 min ; les
# checkpoints "izar_dbg" sont jetables.
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
    --tag izar_dbg \
    --profile
