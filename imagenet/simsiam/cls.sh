#!/bin/bash
#SBATCH --job-name=11-26-simsiam-rot-0.08-200ep
#SBATCH --output=/checkpoint/ljng/latent-noise/cls-log/11-26-simsiam-rot-0.08-200ep.out
#SBATCH --partition=prioritylab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --constraint=volta32gb
#SBATCH --signal=USR1@60
#SBATCH --mem=120G
#SBATCH --open-mode=append
#SBATCH --time 4320
#SBATCH --comment "iclr"

srun --label python main_lincls.py \
              -a resnet50 \
              --dist-url 'tcp://localhost:10001' \
              --multiprocessing-distributed \
              --world-size 1 \
              --rank 0 \
              --lars \
              --pretrained /checkpoint/ljng/latent-noise/tmp/11-26-simsiam-rot-0.08-200ep.pth \
              /datasets01/imagenet_full_size/061417