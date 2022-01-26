# Training

python main.py \
            --dist-url 'tcp://localhost:10001' \
            --multiprocessing-distributed \
            --world-size 1 \
            --rank 0 \
            --rotation 0.08 \
            --checkpoint-dir <checkpoint-dir>

# Evaluation

python main_linear_probe.py \
              -a resnet50 \
              --dist-url 'tcp://localhost:10001' \
              --multiprocessing-distributed \
              --world-size 1 \
              --rank 0 \
              --lars \
              --pretrained /checkpoint/ljng/latent-noise/tmp/11-26-simsiam-rot-0.08-200ep.pth \
              /datasets01/imagenet_full_size/061417