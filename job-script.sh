#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=p100:4
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --account=rrg-wanglab
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=ike.adeyinka@mail.utoronto.ca
#SBATCH --mail-type=ALL


module load cuda cudnn
source tensorflow/bin/activate
python gpuInception_script.py
