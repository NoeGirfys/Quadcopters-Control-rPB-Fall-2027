#!/bin/bash
#SBATCH --job-name=maml_test
#SBATCH --partition=gpu          # file d'attente GPU sur Izar
#SBATCH --qos=debug              # QOS debug : 1h max, haute priorité, gratuit
#SBATCH --gres=gpu:1             # 1 GPU (le test tourne sur CPU, mais la partition gpu en exige un)
#SBATCH --cpus-per-task=2        # CPU
#SBATCH --mem=4G                 # RAM
#SBATCH --time=00:10:00          # le test prend quelques secondes
#SBATCH --output=logs/maml_test_%j.out
#SBATCH --error=logs/maml_test_%j.err

# --- Créer le dossier de logs s'il n'existe pas encore ----------------
mkdir -p logs

# --- Environnement ----------------------------------------------------
module purge
module load gcc python

source venv_MAML_SCITAS/bin/activate

cd $SLURM_SUBMIT_DIR

# --- Test d'équivalence de la boucle externe batchée (optimisation A) -
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
srun python -u training_MAML/test_batched_outer.py
