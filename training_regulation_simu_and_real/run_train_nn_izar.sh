#!/bin/bash
#SBATCH --job-name=cf_nn_reg
#SBATCH --partition=gpu          # GPU queue on Izar
#SBATCH --qos=normal
#SBATCH --gres=gpu:1             # 1 GPU (the rollout is batched over the 27 drones)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00          # 2 short runs of 100 epochs — adjust if needed
#SBATCH --output=logs/cf_nn_%j.out
#SBATCH --error=logs/cf_nn_%j.err

# --- Logs directory ---------------------------------------------------
mkdir -p logs

# --- Environment ------------------------------------------------------
# Training only needs torch + numpy + matplotlib (NO gym-pybullet-drones:
# train_nn_cf_pid.py has its own differentiable dynamics), so any torch venv
# works — we reuse the MAML one here.
module purge
module load gcc python
source venv_MAML_SCITAS/bin/activate

cd $SLURM_SUBMIT_DIR

# --- GPU check --------------------------------------------------------
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Shared hyper-parameters (defaults; terminal_weight / z_weight to be revisited)
COMMON="--epochs 100 --hidden 64 --t_chunk 0.01 --t_sim 2.0 \
        --half_side 0.5 --terminal_weight 50 --z_weight 1.0 \
        --tau_start 0.8 --tau_end 2.0 --eval_plot_every 0"

# --- Run 1: NO observation noise -------------------------------------
echo "=== Training WITHOUT noise ==="
srun python -u training_regulation_simu_and_real/train_nn_cf_pid.py \
    $COMMON --obs_noise_scale 0 --tag noiseless

# --- Run 2: WITH observation noise (domain randomization, scale 2.0) --
echo "=== Training WITH noise (scale 2.0) ==="
srun python -u training_regulation_simu_and_real/train_nn_cf_pid.py \
    $COMMON --obs_noise_scale 2.0 --tag noisy

# --- Overlay the two loss curves for the report ----------------------
python -u training_regulation_simu_and_real/plot_loss_comparison.py \
    --ckpts training_regulation_simu_and_real/trained_cf_pid_T1_ch200_h64_ep100_noiseless.pt \
            training_regulation_simu_and_real/trained_cf_pid_T1_ch200_h64_ep100_noisy.pt \
    --labels "no noise" "noise (scale 2.0)" \
    --out training_regulation_simu_and_real/loss_comparison.png
