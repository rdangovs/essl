#!/bin/bash
#SBATCH --job-name=11-18-barlowtwins-rot-0-1000ep
#SBATCH --output=/checkpoint/ljng/latent-noise/train-log/11-18-barlowtwins-rot-0-1000ep.out
#SBATCH --partition=learnlab
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --constraint=volta32gb
#SBATCH --signal=USR1@60
#SBATCH --mem=120G
#SBATCH --open-mode=append
#SBATCH --time 1440

srun --label python main.py \
            --batch-size 4096 \
            --epochs 1000 \
            --dim 256 \
            --rotation 0 \
            --name 11-18-barlowtwins-rot-0-1000ep