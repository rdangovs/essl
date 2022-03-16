import numpy as np
import torch
import random


def translate_tensor(tensor, input_size=32, nt=2):
    """
    Data augmentation function to enforce periodic boundary conditions.
    Inputs are arbitrarily translated in each dimension
    """
    ndim = len(tensor[0,0, :].shape)
    t = input_size//nt
    t_vec = np.linspace(0, (nt-1)*t, nt).astype(int)
    for i in range(len(tensor)):
        if ndim == 2:
            tensor1 = torch.roll(tensor[i,0, :], (np.random.choice(t_vec),
                                                np.random.choice(t_vec)),
                                 (0, 1))  # translate by random no. of units (0-input_size) in each axis
        elif ndim == 3:
            tensor1 = torch.roll(tensor[i,0, :], (
                np.random.choice(input_size), np.random.choice(input_size), np.random.choice(input_size)), (0, 1, 2))
        else:
            raise
        if i == 0:
            newtensor = tensor1.unsqueeze(0).unsqueeze(0) # add back channel dim and batch dim
        else:
            newtensor = torch.cat((newtensor,tensor1.unsqueeze(0).unsqueeze(0)),dim=0)
    return newtensor

def rotate_tensor(tensor):
    ndim = len(tensor[0,0,:].shape)
    for i in range(len(tensor)):
        tensor1 = tensor[i,0,:]
        rottimes = np.random.choice(4) # 4-fold rotation; rotate by 0, 90, 280 or 270
        rotaxis = np.random.choice(ndim) # axis to rotate [0,1], [1,0] in 2D (double count negative rot is ok) and [0,1], [1,2], [2,0] in 3D (negative rotation covered by k = 3)
        tensor1 = torch.rot90(tensor1,k=rottimes,dims=[rotaxis,(rotaxis+1)%(ndim)])
        if i == 0:
            newtensor = tensor1.unsqueeze(0).unsqueeze(0)
        else:
            newtensor = torch.cat((newtensor,tensor1.unsqueeze(0).unsqueeze(0)),dim=0)
    return newtensor

def flip_tensor(tensor):
    ndim = len(tensor[0,0,:].shape)
    for i in range(len(tensor)):
        if ndim == 2:
            flipaxis = np.random.choice(3,1)
            flipaxis = [] if flipaxis.item() == 2 else list(flipaxis)
            # flipaxis = np.random.choice([[0],[1],[]]) # flip hor, ver, or None (dont include Diagonals = flip + rot90)
        elif ndim == 3:
            flipaxis = np.random.choice(4,1)
            flipaxis = [] if flipaxis.item() == 3 else list(flipaxis)
            # flipaxis = np.random.choice([[0],[1],[2],[]]) # flip x, y, z or None (dont include Diagonals = flip + rotate)
        tensor1 = torch.flip(tensor[i,0,:],flipaxis)
        if i == 0:
            newtensor = tensor1.unsqueeze(0).unsqueeze(0)
        else:
            newtensor = torch.cat((newtensor,tensor1.unsqueeze(0).unsqueeze(0)),dim=0)
    return newtensor

def scale_tensor(tensor, mineps = 1, maxeps = 20):
    ndim = len(tensor[0,0,:].shape)
    # count = 0
    for i in range(len(tensor)):
        tensornew1 = torch.zeros_like(tensor[i,0,:])
        tensor1 = tensor[i,0,:]*(maxeps-mineps)+mineps # unstandardize input
        while not (torch.min(tensornew1)>mineps and torch.max(tensornew1)<maxeps): # we normalized input to be in 0 and 1
            factor = -random.uniform(-2,0) # uniform includes low but excludes high. want to include 1 but exclude 0
            tensornew1 = tensor1*factor
        if i == 0:
            newtensor = tensornew1.unsqueeze(0).unsqueeze(0)
        else:
            newtensor = torch.cat((newtensor,tensornew1.unsqueeze(0).unsqueeze(0)),dim=0)
    return (newtensor-mineps)/(maxeps-mineps) # standardize back input

def translate_cont_tensor(tensor,input_size=32):
    '''Data augmentation function to enforce periodic boundary conditions. Inputs are arbitrarily translated in each dimension'''
    ndim = len(tensor[0,0,:].shape)
    for i in range(len(tensor)):
        if ndim == 2:
            tensor1 = torch.roll(tensor[i,0,:],(np.random.choice(input_size),np.random.choice(input_size)),(0,1)) # translate by random no. of units (0-input_size) in each axis
        elif ndim == 3:
            tensor1 = torch.roll(tensor[i,0,:],(np.random.choice(input_size),np.random.choice(input_size),np.random.choice(input_size)),(0,1,2))
        if i == 0:
            newtensor = tensor1.unsqueeze(0).unsqueeze(0) # add back channel dim and batch dim
        else:
            newtensor = torch.cat((newtensor,tensor1.unsqueeze(0).unsqueeze(0)),dim=0)
    return newtensor

def get_aug(x,translate=False,flip=False,rotate=False,scale=False, nt=2, p=1,translate_cont=False,pg_uniform=False):
    if not translate_cont:
        if translate:
            if np.random.choice(int(1/p)) == 0:
                x = translate_tensor(x,nt=nt)
    else:
        if np.random.choice(int(1/p)) == 0:
            x = translate_cont_tensor(x)
    if pg_uniform:
        if np.random.choice(int(1/p)) == 0:
            x = get_pg_uni(x, dim = 2)
    else:
        if flip:
            if np.random.choice(int(1/p)) == 0:
                x = flip_tensor(x)
        if rotate:
            if np.random.choice(int(1/p)) == 0:
                x = rotate_tensor(x)
    if scale:
        if np.random.choice(int(1/p)) == 0:
            x = scale_tensor(x)
    return x


def transform_and_get_labels(x,translate,flip,rotate,nt=2, uc = 32):
    if translate+flip+rotate != 1:
        raise ValueError("only 1 transformation allowed")
    if translate:
        x, y = trans_and_get_labels(x, nt=nt, input_size=uc)
    if flip:
        x, y = flip_and_get_labels(x)
    if rotate:
        x, y = rot4_and_get_labels(x)
    return x, y

def rot4_and_get_labels(x):
    bsz = x.shape[0]
    new_bsz = 4*bsz
    # rotate images all 4 ways at once
    new_x = torch.zeros([new_bsz, x.shape[1], x.shape[2], x.shape[3]])
    new_y = torch.zeros([new_bsz]).long()
    for i in range(4):
        new_x[i*bsz:(i+1)*bsz] = torch.rot90(x,i,[2,3])
        new_y[i*bsz:(i+1)*bsz] = i
    return new_x, new_y

def trans_and_get_labels(x, nt=2, input_size=32):
    bsz = x.shape[0]
    new_bsz = (nt**2)*bsz # number of classes is nt^2
    t = input_size//nt
    t_vec = torch.linspace(0, (nt-1)*t, nt).long()
    tlist = []
    for i in t_vec:
        for j in t_vec:
            tlist.append((i,j))
    # trans images all discrete ways at once
    new_x = torch.zeros([new_bsz, x.shape[1], x.shape[2], x.shape[3]])
    new_y = torch.zeros([new_bsz]).long()
    for i in range(nt**2):
        new_x[i*bsz:(i+1)*bsz] = torch.roll(x, tlist[i], (2,3))
        new_y[i*bsz:(i+1)*bsz] = i
    return new_x, new_y

def flip_and_get_labels(x):
    bsz = x.shape[0]
    new_bsz = 4*bsz
    tlist = [[],[2],[3],[2,3]]

    new_x = torch.zeros([new_bsz, x.shape[1], x.shape[2], x.shape[3]])
    new_y = torch.zeros([new_bsz]).long()
    for i in range(4):
        new_x[i*bsz:(i+1)*bsz] = torch.flip(x, tlist[i])
        new_y[i*bsz:(i+1)*bsz] = i
    return new_x, new_y

if __name__ == 'main':
    pass
