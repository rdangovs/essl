#!/bin/bash 
#SBATCH --job-name=1-26-essl-simclr-0.4-100ep
#SBATCH --output=/checkpoint/ljng/essl/train-log/1-26-essl-simclr-0.4-100ep.out 
#SBATCH --partition=learnlab
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --constraint=volta32gb
#SBATCH --signal=USR1@60
#SBATCH --mem=120G
#SBATCH --open-mode=append
#SBATCH --time 4320
srun --label python main.py \
    --data /datasets01/imagenet_full_size/061417 \
    --epochs 100 \
    --rotation 0.4 \
	--name /checkpoint/ljng/essl/1-26-essl-simclr-0.4-100ep