#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=def-wanglab-ab
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=ike.adeyinka@mail.utoronto.ca
#SBATCH --mail-type=ALL


module load cuda cudnn
source tensorflow/bin/activate
python gpuInception_script.py
