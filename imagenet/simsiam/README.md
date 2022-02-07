# Equivariant SimSiam

This sub-repo is an implementation of E-SimSiam, the E-SSL version of SimSiam. 
We based our code on the official implementation of [SimSiam](https://github.com/facebookresearch/simsiam).
You can use the training and evaluation scripts to obtain a SimSiam baseline and our E-SimSiam improvement.

### Training 
```
python main.py \
            --dist-url 'tcp://localhost:10001' \
            --multiprocessing-distributed \
            --world-size 1 \
            --rank 0 \
            --rotation 0.08 \  # 0.0 for baseline 
            --data <data-dir> \
            --checkpoint-dir <checkpoint-dir>
```

### Evaluation 
```
python main_linear_probe.py \
              -a resnet50 \
              --dist-url 'tcp://localhost:10001' \
              --multiprocessing-distributed \
              --world-size 1 \
              --rank 0 \
              --lars \
              --pretrained <checkpoint> \
              --data <data-dir>
```

### Pretrained Models

<table>
  <tr>
    <th>experiment</th>
    <th>epochs</th>
    <th>acc1</th>
    <th colspan="3">download</th>
  </tr>
  <tr>
    <td>SimSiam (baseline)</td>
    <td>100</td>
    <td>67.9%</td>
    <td><a href="https://www.dropbox.com/s/xkbzpujvuyqeri8/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/d59gha1xc9cqakf/1-31-essl-simsiam-0.out">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/6i5wqm4hr657i8c/linear_probe.out">val logs</a></td>
  </tr>
  <tr>
    <td>E-SimSiam (ours)</td>
    <td>100</td>
    <td>68.5%</td>
    <td><a href="https://www.dropbox.com/s/gimwx0eb7lbi66u/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/trqdtwgukou99lq/1-31-essl-simsiam-0.08.out">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/gqu8ce712yk2dh2/linear_probe.out">val logs</a></td>
  </tr>
</table>

Settings for the above: 8 NVIDIA V100 GPUs, CUDA 10.2, PyTorch 1.10.1, Torchvision 0.11.2.