#!/bin/bash
#SBATCH --job-name=11-13-barlowtwins-rot-8-200ep-lr-1.0
#SBATCH --output=/checkpoint/ljng/latent-noise/cls-log/11-13-barlowtwins-rot-8-200ep-lr-1.0.out
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --constraint=volta32gb
#SBATCH --signal=USR1@60
#SBATCH --mem=120G
#SBATCH --open-mode=append
#SBATCH --time 1440

srun --label python evaluate.py \
        --name 11-13-barlowtwins-rot-8-200ep \
        --lr-classifier 1.0
