#!/bin/bash
#SBATCH --time=99:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
module load Python/3.6.4-foss-2018a
python ./python_gpu.py