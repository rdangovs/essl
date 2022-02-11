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
    <th colspan="4">download</th>
  </tr>
  <tr>
    <td>SimCLR (baseline)</td>
    <td>100</td>
    <td> - </td>
    <td> 67.6% </td>
    <td><a href="https://www.dropbox.com/s/vnz581qrwz03c2p/resnet50.pth">ResNet-50</a></td> 
    <td><a href="https://www.dropbox.com/s/osys5247ucyn331/checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://www.dropbox.com/s/x7tprausdzahqxl/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/vxco7lnecz6yxps/linear_probe.out">val logs</a></td>
  </tr>
  <tr>
    <td>E-SimCLR (ours)</td>
    <td>100</td>
    <td>(0.05, 0.14) </td>
    <td>68.2%</td>
    <td><a href="https://www.dropbox.com/s/5n2hnqiif7zbvx3/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/48yyalyerzsnr1i/checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://www.dropbox.com/s/0zlvi214gdir70c/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/65v1aq91q765j0x/linear_probe.out">val logs</a></td>
  </tr>
   <tr>
    <td>E-SimCLR (ours)</td>
    <td>100</td>
    <td>(0.08, 1.0) </td>
    <td>68.2%</td>
    <td><a href="">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/6a3ldim0s0s18fa/checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://www.dropbox.com/s/86t7do3f7mwkfgw/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/epj5rgm3g89fm83/linear_probe.out">val logs</a></td>
  </tr>
  <tr>
    <td>SimCLR (baseline)</td>
    <td>200</td>
    <td> - </td>
    <td> 69.7% </td>
    <td><a href="https://www.dropbox.com/s/prqz8iu2c3ngtfj/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/pwair41fai2lhiv/checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://www.dropbox.com/s/n36r4tsybbhm8aa/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/kjr1j7paxhx5enw/linear_probe.out">val logs</a></td>
  </tr>
  <tr>
    <td>E-SimCLR (ours)</td>
    <td>200</td>
    <td>(0.05, 0.14) </td>
    <td>70.2%</td>
    <td><a href="https://www.dropbox.com/s/82j08xb2i3duhuq/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/n5wh5bvtp9cuzvw/checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://www.dropbox.com/s/1gzgyy7i75me6mq/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/2mf5q3f85fn09kl/linear_probe.out">val logs</a></td>
  </tr>
   <tr>
    <td>E-SimCLR (ours)</td>
    <td>200</td>
    <td>(0.08, 1.0) </td>
    <td>70.2%</td>
    <td><a href="https://www.dropbox.com/s/drcpwbzxc1ew4df/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/nu05zb4jzd9hxmp/checkpoint.pth">full checkpoint</a></td>    
    <td><a href="https://www.dropbox.com/s/3jihgup4bxrw1m1/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/ni0d0fnh5ra80cq/linear_probe.out">val logs</a></td>
  </tr>
  <tr>
    <td>SimCLR (baseline)</td>
    <td>300</td>
    <td> - </td>
    <td> 70.4% </td>
    <td><a href="https://www.dropbox.com/s/n15o23q4tdyt484/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/js0h73czn486sa0/checkpoint.pth">full checkpoint</a></td>    
    <td><a href="https://www.dropbox.com/s/q1n11j4h4nen7ji/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/ftj64k6rx6u3ch9/linear_probe.out">val logs</a></td>
  </tr>
  <tr>
    <td>E-SimCLR (ours)</td>
    <td>300</td>
    <td>(0.05, 0.14) </td>
    <td>71.3%</td>
    <td><a href="https://www.dropbox.com/s/2jl84kdfossu9qy/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/adpifeqqaykvrqx/checkpoint.pth">full checkpoint</a></td>    
    <td><a href="https://www.dropbox.com/s/zlcfuf935vtcg7i/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/efnnle4j6xnejkb/linear_probe.out">val logs</a></td>
  </tr>
   <tr>
    <td>E-SimCLR (ours)</td>
    <td>300</td>
    <td>(0.08, 1.0) </td>
    <td>71.1%</td>
    <td><a href="https://www.dropbox.com/s/gk61bwhzktt607b/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/r4ntxt9spwc8t52/checkpoint.pth">full checkpoint</a></td>    
    <td><a href="https://www.dropbox.com/s/qvwsncpmgb51gpi/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/77y8behax05bpbo/linear_probe.out">val logs</a></td>
  </tr>
  <tr>
    <td>SimCLR (baseline)</td>
    <td>800</td>
    <td>- </td>
    <td>71.9%</td>
    <td><a href="https://www.dropbox.com/s/dvol1bdpzdu87bl/10-16-simclr-rot-0.0-800ep-resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/gvof9uovvn32b6e/11-26-simclr-rot-0.0-800ep-lr-0.3.out">train and val logs</a></td>
  </tr>
   <tr>
    <td>E-SimCLR (ours)</td>
    <td>800</td>
    <td>(0.05, 0.14) </td>
    <td>72.5%</td>
    <td><a href="https://www.dropbox.com/s/0vtxnygl8sywtgk/11-26-simclr-rot-0.4-800ep-resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/q3t4qj32nwn6uku/11-26-simclr-rot-0.4-800ep-lr-0.3.out">train and val logs</a></td>
  </tr>
</table>

Settings for the above: 32 NVIDIA V100 GPUs for training (8 GPUs for evaluation), CUDA 10.2, PyTorch 1.10.1, Torchvision 0.11.2.
