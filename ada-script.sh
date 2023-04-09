#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --job-name="inception_test"
#SBATCH --gres=gpu:4
#SBATCH --account=def-wanglab-ab
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=ike.adeyinka@mail.utoronto.ca
#SBATCH --mail-type=ALL


module load cuda cudnn
source tensorflow/bin/activate
python gpuInception_script.py
