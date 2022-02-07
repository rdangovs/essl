# Equivariant SimCLR

This sub-repo is an implementation of E-SimCRL, the E-SSL version of SimCLR.
You can use the training and evaluation scripts to obtain a SimCLR baseline and our E-SimCLR improvement.

### Training
```
python main.py --data <path-to-data> --checkpoint-dir <checkpoint-dir> --rotation 0.4   # 0 for baseline
```

### Evaluation
```
python linear_probe.py --data <path-to-data> --pretrained <path-to-pretrained>
```

### Pretrained Models

<table>
  <tr>
    <th>experiment</th>
    <th>epochs</th>
    <th>small crop scale</th>
    <th>acc1</th>
    <th colspan="3">download</th>
  </tr>
  <tr>
    <td>SimCLR (baseline)</td>
    <td>100</td>
    <td> - </td>
    <td> </td>
    <td><a href="">ResNet-50</a></td>
    <td><a href="">train logs</a></td>
    <td><a href="">val logs</a></td>
  </tr>
  <tr>
    <td>E-SimCLR (ours)</td>
    <td>100</td>
    <td>(0.05, 0.14) </td>
    <td>%</td>
    <td><a href="">ResNet-50</a></td>
    <td><a href="">train logs</a></td>
    <td><a href="">val logs</a></td>
  </tr>
   <tr>
    <td>E-SimCLR (ours)</td>
    <td>100</td>
    <td>(0.08, 1.0) </td>
    <td>%</td>
    <td><a href="">ResNet-50</a></td>
    <td><a href="">train logs</a></td>
    <td><a href="">val logs</a></td>
  </tr>
  <tr>
    <td>SimCLR (baseline)</td>
    <td>200</td>
    <td> - </td>
    <td> </td>
    <td><a href="">ResNet-50</a></td>
    <td><a href="">train logs</a></td>
    <td><a href="">val logs</a></td>
  </tr>
  <tr>
    <td>E-SimCLR (ours)</td>
    <td>200</td>
    <td>(0.05, 0.14) </td>
    <td>%</td>
    <td><a href="">ResNet-50</a></td>
    <td><a href="">train logs</a></td>
    <td><a href="">val logs</a></td>
  </tr>
   <tr>
    <td>E-SimCLR (ours)</td>
    <td>200</td>
    <td>(0.08, 1.0) </td>
    <td>%</td>
    <td><a href="">ResNet-50</a></td>
    <td><a href="">train logs</a></td>
    <td><a href="">val logs</a></td>
  </tr>
  <tr>
    <td>SimCLR (baseline)</td>
    <td>300</td>
    <td> - </td>
    <td> </td>
    <td><a href="">ResNet-50</a></td>
    <td><a href="">train logs</a></td>
    <td><a href="">val logs</a></td>
  </tr>
  <tr>
    <td>E-SimCLR (ours)</td>
    <td>300</td>
    <td>(0.05, 0.14) </td>
    <td>%</td>
    <td><a href="">ResNet-50</a></td>
    <td><a href="">train logs</a></td>
    <td><a href="">val logs</a></td>
  </tr>
   <tr>
    <td>E-SimCLR (ours)</td>
    <td>300</td>
    <td>(0.08, 1.0) </td>
    <td>%</td>
    <td><a href="">ResNet-50</a></td>
    <td><a href="">train logs</a></td>
    <td><a href="">val logs</a></td>
  </tr>
  <tr>
    <td>SimCLR (baseline)</td>
    <td>800</td>
    <td>- </td>
    <td>%</td>
    <td><a href="">ResNet-50</a></td>
    <td><a href="">train logs</a></td>
    <td><a href="">val logs</a></td>
  </tr>
   <tr>
    <td>E-SimCLR (ours)</td>
    <td>800</td>
    <td>(0.05, 0.14) </td>
    <td>%</td>
    <td><a href="">ResNet-50</a></td>
    <td><a href="">train logs</a></td>
    <td><a href="">val logs</a></td>
  </tr>
</table>

Settings for the above: 32 NVIDIA V100 GPUs for training (8 GPUs for evaluation), CUDA 10.2, PyTorch 1.10.1, Torchvision 0.11.2.
