#!/bin/bash
#SBATCH --job-name=maml_test_pyb
#SBATCH --partition=gpu          # file d'attente GPU sur Izar
#SBATCH --qos=debug              # QOS debug : 1h max, haute priorité, gratuit
#SBATCH --gres=gpu:1             # 1 GPU (la partition gpu en exige un ; le test tourne sur CPU)
#SBATCH --cpus-per-task=4        # CPU (PyBullet)
#SBATCH --mem=8G                 # RAM
#SBATCH --time=00:20:00          # deux simus PyBullet courtes
#SBATCH --output=logs/test_pyb_%j.out
#SBATCH --error=logs/test_pyb_%j.err

mkdir -p logs

module purge
module load gcc python
source venv_MAML_SCITAS/bin/activate
cd $SLURM_SUBMIT_DIR

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"

# ====================================================================
#  Checkpoints (472 epochs, mêmes tâches / x0)
# ====================================================================
MAML_CKPT=training_MAML/maml_linearized_h64_o1_n50_izar_uniform_tasks_ep472.pt
BASE_CKPT=training_MAML/baseline_linearized_h64_n50_izar_uniform_tasks_ep472.pt

# ====================================================================
#  Conditions de test communes — À ÉDITER selon le scénario voulu.
#  Définies une seule fois => MAML et baseline les partagent à l'identique.
# ====================================================================
TARGET_DX=0.03      # offset de la masse, dx [m]
TARGET_DY=0.01      # offset de la masse, dy [m]
TARGET_DZ=0.0       # offset de la masse, dz [m]
MASS=0.010          # masse additionnelle [kg]

TARGET_POS="0 0 1"  # cible de régulation du NN [m]
INIT_POS="0 0 1"    # position initiale du drone [m]
INIT_RPY="0 0 0"    # roll/pitch/yaw initial [rad]
DURATION=8.0        # durée de la simu [s]
DEVICE=cpu          # même device pour les deux => adaptation identique

COMMON="--target-dx $TARGET_DX --target-dy $TARGET_DY --target-dz $TARGET_DZ \
        --mass $MASS --target-pos $TARGET_POS --init-pos $INIT_POS \
        --init-rpy $INIT_RPY --duration $DURATION --device $DEVICE"

# ====================================================================
#  1) Contrôleur MAML
# ====================================================================
echo "=== TEST MAML ==="
srun python -u training_MAML/test_maml_pybullet.py \
    --weights $MAML_CKPT $COMMON

# ====================================================================
#  2) Baseline — adaptée avec les MÊMES hyperparamètres que MAML
#     (--maml-ckpt fournit n_inner_steps / lr_inner / inner_grad_clip)
# ====================================================================
echo "=== TEST BASELINE ==="
srun python -u training_MAML/test_maml_pybullet.py \
    --weights $BASE_CKPT --maml-ckpt $MAML_CKPT $COMMON

echo "Done. Plots: *_test_phaseB.png dans training_MAML/"
