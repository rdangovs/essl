
import torch.nn as nn
import torch
import math
import numpy as np

class Encoder(nn.Module):

    def __init__(self, Lv, ks):
        super(Encoder, self).__init__()
        self.enc_block2d = nn.Sequential(
            nn.Conv2d(1, Lv[0], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
            nn.BatchNorm2d(Lv[0]),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            # nn.Dropout(p=0.2),
            nn.Conv2d(Lv[0], Lv[1], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
            nn.BatchNorm2d(Lv[1]),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            # nn.Dropout(p=0.2),
            nn.Conv2d(Lv[1], Lv[2], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
            nn.BatchNorm2d(Lv[2]),
            nn.ReLU(),
            nn.MaxPool2d(4,4)
            )
        self.fcpart = nn.Sequential(
            nn.Linear(Lv[2] * 1 * 1, Lv[3]),
            nn.ReLU(),
            nn.Linear(Lv[3], Lv[4]),
            )
        self.Lv = Lv

    def forward(self, x):
        x = self.enc_block2d(x)
        x = x.view(-1, self.Lv[2] * 1 * 1)
        x = self.fcpart(x)
        return x


class Projector(nn.Module):

    def __init__(self, Lvpj, hidden_dim, bnorm=False, depth=2):
        super(Projector, self).__init__()
        nlayer = [nn.BatchNorm1d(Lvpj[0])] if bnorm else []
        list_layers = [nn.Linear(hidden_dim, Lvpj[0])] + nlayer + [nn.ReLU()]
        for _ in range(depth-2):
            list_layers += [nn.Linear(Lvpj[0], Lvpj[0])] + nlayer + [nn.ReLU()]
        list_layers += [nn.Linear(Lvpj[0],Lvpj[1])]
        self.proj_block = nn.Sequential(*list_layers)

    def forward(self, x):
        x = self.proj_block(x)
        return x

class Classifier(nn.Module):
    def __init__(self, Lvpj, in_dim, linear = False, bnorm=False, nrot = 4, depth=2):
        super(Classifier, self).__init__()
        if not linear:
            nlayer = [nn.BatchNorm1d(Lvpj[0])] if bnorm else []
            list_layers = [nn.Linear(in_dim, Lvpj[0])] + nlayer + [nn.ReLU()]
            for _ in range(depth-2):
                list_layers += [nn.Linear(Lvpj[0], Lvpj[0])] + nlayer + [nn.ReLU()]
            list_layers += [nn.Linear(Lvpj[0],nrot)]
        else:
            list_layers = [nn.Linear(in_dim, nrot)]

        self.clas_block = nn.Sequential(*list_layers)

    def forward(self, x):
        x = self.clas_block(x)
        return x


class PredictorDOS(nn.Module):

    def __init__(self, num_dos, hidden_dim, Lvp):
        super(PredictorDOS, self).__init__()

        self.num_dos = num_dos
        self.hidden_dim = hidden_dim
        self.Lvp = Lvp
        self.DOSband = self.fc_block()

    def fc_block(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.Lvp[0]),
            nn.ReLU(),
            nn.Linear(self.Lvp[0], self.Lvp[1]),
            nn.ReLU(),
            nn.Linear(self.Lvp[1], self.Lvp[2]),
            nn.ReLU(),
            nn.Linear(self.Lvp[2], self.num_dos),
            )

    def forward(self, x):
        x = self.DOSband(x)
        return x

class Net(nn.Module):
    def __init__(self,Lv,ks,Lvp):
        super(Net, self).__init__()
        self.enc = Encoder(Lv,ks)
        self.predictor = PredictorDOS(400,Lv[-1],Lvp)
    def forward(self, x):
      x = self.enc(x)
      x = self.predictor(x)
      return x

# define new sub module classes to keep the same state dict keys
class Enc(nn.Module):
    def __init__(self,Lv,ks):
        super(Enc, self).__init__()
        self.enc= Encoder(Lv,ks)
    def forward(self, x):
      x = self.enc(x)
      return x

class Proj(nn.Module):
    def __init__(self,Lvpj,latent_dim, bnorm=False, depth=2):
        super(Proj, self).__init__()
        self.projector = Projector(Lvpj,latent_dim, bnorm=bnorm, depth=depth)
    def forward(self, x):
      x = self.projector(x)
      return x

class Clas(nn.Module):
    def __init__(self,Lvpj,latent_dim, linear = False, bnorm = False, nrot = 4, depth=2):
        super(Clas, self).__init__()
        self.classifier = Classifier(Lvpj,latent_dim, linear, bnorm, nrot, depth)
    def forward(self, x):
      x = self.classifier(x)
      return x


if __name__ == '__main__':
    pass
