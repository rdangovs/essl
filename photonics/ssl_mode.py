import copy
import random
import torch
from torch import nn
import torch.nn.functional as F

from phc_models import Enc, Proj, Clas # model classes
from utils import NTXentLoss

class SimCLRwEE(nn.Module):

    def __init__(self, Lv,Lvpj,ks,device,batchsize_ptxt,temperature,
                use_projector = False, nrot=4, depth=3,
                scdepth=3, eeloss='xent',relative_ee= False):
        super().__init__()
        self.NTXent = NTXentLoss(device,batchsize_ptxt,temperature)
        self.projector = Proj(Lvpj,Lv[-1], bnorm=True, depth=scdepth)
        self.use_projector = use_projector
        self.encoder = Enc(Lv,ks)
        self.bsz = batchsize_ptxt

        if relative_ee:
            self.classifier = Clas(Lvpj,Lv[-1]*2,linear = False, bnorm=True, nrot=nrot,depth=depth) # we use same projector for RotNet and SimCLR
        else:
            self.classifier = Clas(Lvpj,Lv[-1],linear = False, bnorm=True, nrot=nrot,depth=depth) # we use same projector for RotNet and SimCLR
        self.ee_loss = eeloss

    def forward(self, x1, x2, xrot, labels):

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        zrot = self.encoder(xrot) # size = [2*bsz, pj_dim]
        if zrot.shape[0] == self.bsz*2:
            zrot1, zrot2 = torch.split(zrot, [self.bsz, self.bsz], dim=0)
            zrot = torch.cat([zrot1,zrot2],dim=-1) #size = [bsz,2*pj_dim]

        if self.use_projector:
            z1 = self.projector(z1)
            z2 = self.projector(z2)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        loss = self.NTXent(z1, z2)
        zrot = self.classifier(zrot)
        if self.ee_loss == 'xent':
            loss2 = F.cross_entropy(zrot, labels)
        elif self.ee_loss == 'l1':
            loss2 = F.l1_loss(zrot, labels.view(-1,1))
        elif self.ee_loss == 'mse':
            loss2 = F.mse_loss(zrot, labels.view(-1,1))
        return loss, loss2

class SimCLR(nn.Module):

    def __init__(self, Lv,Lvpj,ks,device,batchsize_ptxt,temperature,
    use_projector = False, scdepth=3):
        super().__init__()
        self.NTXent = NTXentLoss(device,batchsize_ptxt,temperature)
        self.projector = Proj(Lvpj,Lv[-1], bnorm=True, depth=scdepth)
        self.use_projector = use_projector

        self.encoder = Enc(Lv,ks)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        if self.use_projector:
            z1 = self.projector(z1)
            z2 = self.projector(z2)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        loss = self.NTXent(z1, z2)
        return loss

if __name__ == "__main__":
    pass
