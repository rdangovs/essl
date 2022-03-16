E-SSL: Photonics
---------------------------------------------------------------

<p align="center">
  <img width="700" alt="phc" src="https://user-images.githubusercontent.com/19780421/158633605-825ced70-4488-484a-9bc8-2b495082bd16.png">
</p>

Code for SimCLR baselines and E-SimCLR for a regression task in photonics science, using two photonic crystals datasets: "Blob" and "Gpm".

### Prerequisites

Please download the ["Blob" and "Gpm" photonic crystals datasets](https://www.dropbox.com/sh/vvrszis4zpfniok/AADH-98EwGC6qfIULr_xPNfUa?dl=0). For more information on generating the datasets, refer to our [previous work](https://github.com/clott3/SIB-CL) 
(simulations and calculations use the [Python MPB package](https://mpb.readthedocs.io/en/latest/) and MATLAB).

The code was last tested with PyTorch 1.8.0 on a single V100 32GB GPU.

### Training and Evaluation

To run the SimCLR baselines:

```
python main.py --dataset=blob --ssl_mode=simclr
python main.py --dataset=gpm --ssl_mode=simclr
```

To run the poor SimCLR + Transform baseline (when invariance is encouraged):

```
python main.py --dataset=blob --ssl_mode=simclr_trans
python main.py --dataset=gpm --ssl_mode=simclr_trans
```

To run the E-SimCLR improvement (when non-trivial equivariance is encouraged):

```
python main.py --dataset=blob --ssl_mode=simclr_ee
python main.py --dataset=gpm --ssl_mode=simclr_ee
```

