# Equivariant Barlow Twins 

This sub-repo is an implementation of E-BT, the E-SSL version of Barlow Twins (BT).
We based our code on the official implementation of [Barlow Twins](https://github.com/facebookresearch/barlowtwins).
You can use the training and evaluation scripts to obtain a BT baseline and our E-BT improvement.

### Training
```
python main.py --data <path-to-data> --checkpoint-dir <checkpoint-dir> --rotation 8  # 0 for baseline
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
    <td>BT (baseline)</td>
    <td>100</td>
    <td> - </td>
    <td> 66.7% </td>
    <td><a href="https://www.dropbox.com/s/kr0vuwgh824z74v/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/zyjacireqrlyj1j/checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://www.dropbox.com/s/a2y2as699egx9tq/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/7m575itcnkrohse/linear_probe.out">val logs</a></td>
  </tr>
  <tr>
    <td>E-BT (ours)</td>
    <td>100</td>
    <td>(0.05, 0.14) </td>
    <td>67.5%</td>
    <td><a href="https://www.dropbox.com/s/n59l1xsz8ze45f2/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/15t8omi4w218vfl/checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://www.dropbox.com/s/0ss9k3y41jo3eb1/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/2mugqmagg4uq1vy/linear_probe.out">val logs</a></td>
  </tr>
   <tr>
    <td>E-BT (ours)</td>
    <td>100</td>
    <td>(0.08, 1.0) </td>
    <td>67.3%</td>
    <td><a href="https://www.dropbox.com/s/tzxhhie8jz2imlm/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/q8wquss96nya6iq/checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://www.dropbox.com/s/gyby21qqyu72a50/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/6qnc4zz2ph567f3/linear_probe.out">val logs</a></td>
  </tr>
  <tr>
    <td>BT (baseline)</td>
    <td>200</td>
    <td> - </td>
    <td> 69.8% </td>
    <td><a href="https://www.dropbox.com/s/10bx6rfzbheyecs/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/vq6gh19hd909oan/checkpoint.pth">full checkpoint</a></td>    
    <td><a href="https://www.dropbox.com/s/ubcx5m30kr0459c/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/ivqoco29dcr3uxi/linear_probe.out">val logs</a></td>
  </tr>
  <tr>
    <td>E-BT (ours)</td>
    <td>200</td>
    <td>(0.05, 0.14) </td>
    <td>70.9%</td>
    <td><a href="https://www.dropbox.com/s/uexnb6aug91v9xa/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/ymt3huvgupzes17/checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://www.dropbox.com/s/iege8quvghc7fgd/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/hdsc8mustvoxijr/linear_probe.out">val logs</a></td>
  </tr>
   <tr>
    <td>E-BT (ours)</td>
    <td>200</td>
    <td>(0.08, 1.0) </td>
    <td>70.6%</td>
    <td><a href="https://www.dropbox.com/s/07nftexd405psno/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/ucu1xbwvsyzx0l9/checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://www.dropbox.com/s/jgb4olumk4u0dh2/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/6szqzmyfqzped97/linear_probe.out">val logs</a></td>
  </tr>
  <tr>
    <td>BT (baseline)</td>
    <td>300</td>
    <td> - </td>
    <td> 70.9% </td>
    <td><a href="https://www.dropbox.com/s/nt0u90z9t1s02yn/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/5c5tmvy8mshnngf/checkpoint.pth">full checkpoint</a></td>    
    <td><a href="https://www.dropbox.com/s/kacdhh1kqhzkznn/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/zcxfpwt3rn5tot0/linear_probe.out">val logs</a></td>
  </tr>
  <tr>
    <td>E-BT (ours)</td>
    <td>300</td>
    <td>(0.05, 0.14) </td>
    <td>71.8%</td>
    <td><a href="https://www.dropbox.com/s/o8f992wijz594gy/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/12uxwxz912mfzzz/checkpoint.pth">full checkpoint</a></td>    
    <td><a href="https://www.dropbox.com/s/nxopboudhqq4sje/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/m21httg21guhzgl/linear_probe.out">val logs</a></td>
  </tr>
   <tr>
    <td>E-BT (ours)</td>
    <td>300</td>
    <td>(0.08, 1.0) </td>
    <td>71.8%</td>
    <td><a href="https://www.dropbox.com/s/sc6emjmg2ytmf9z/resnet50.pth">ResNet-50</a></td>
    <td><a href="https://www.dropbox.com/s/65n5tszxz0p449a/checkpoint.pth">full checkpoint</a></td>
    <td><a href="https://www.dropbox.com/s/d697cffn0j2mt9w/stats.txt">train logs</a></td>
    <td><a href="https://www.dropbox.com/s/kl4m432op6xb18u/linear_probe.out">val logs</a></td>
  </tr>
</table>

Settings for the above: 32 NVIDIA V100 GPUs for training (8 GPUs for evaluation), CUDA 10.2, PyTorch 1.10.1, Torchvision 0.11.2.
