#!/bin/bash
#SBATCH --job-name=11-26-simsiam-rot-0.08-200ep
#SBATCH --output=/checkpoint/ljng/latent-noise/train-log/11-26-simsiam-rot-0.08-200ep.out
#SBATCH --partition=learnlab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --constraint=volta32gb
#SBATCH --signal=USR1@60
#SBATCH --mem=120G
#SBATCH --open-mode=append
#SBATCH --time 4320

srun --label python main_simsiam.py \
            --dist-url 'tcp://localhost:10001' \
            --multiprocessing-distributed \
            --world-size 1 \
            --rank 0 \
            --rotation 0.08 \
            --epochs 200 \
            --resume yes \
            --name 11-26-simsiam-rot-0.08-200ep