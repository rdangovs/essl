E-SSL: CIFAR-10
---------------------------------------------------------------

<p align="center">
  <img width="500" alt="cifar10" src="https://user-images.githubusercontent.com/19780421/147094896-0b1166d5-742a-4a87-bad0-9e04741256bc.png">
</p>

Minimal code for E-SimCLR and E-SimSiam and strong baselines for SimCLR and SimSiam on CIFAR-10.

### Prerequisites

The code was last tested with PyTorch 1.10.0 and Torchvision 0.11.1 on a single V100 32GB GPU.

### Training and Evaluation

To run the SimCLR baseline:

```
python main.py --loss=simclr --lr=0.03 --lmbd=0.0
```

To run the E-SimCLR improvement:

```
python main.py --loss=simclr --lr=0.06 --lmbd=0.4
```

To run the SimSiam baseline:

```
python main.py --loss=simsiam --lr=0.03 --lmbd=0.0
```

To run the E-SimSiam improvement:

```
python main.py --loss=simsiam --lr=0.06 --lmbd=0.4
```

<p align="center">
  <img width="500" alt="cifar10" src="https://user-images.githubusercontent.com/19780421/147096631-eb06b429-ea1c-4420-88ac-be4751af05ea.png">
</p>

Notes:

* We recommend using `--fp16` for speeding up the experiments. Further, substantial speedups can be achieved
  with [FFCV](https://github.com/libffcv/ffcv).

