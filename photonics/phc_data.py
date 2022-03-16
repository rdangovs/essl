import torch
from torch.utils import data
from torchvision.transforms import transforms
import numpy as np
import h5py
import os
import math
import matplotlib.pyplot as plt

class PhC2D(data.Dataset):

    def __init__(self, path_to_h5_dir, trainsize, validsize=0, testsize=2000,
        predict = 'DOS', mode = 'ELdiff', targetmode = 'none', split='train',
        ldos = 0, udos = 400, ftsetind = None, dataset='blob'):
        """
        :param path_to_h5_dir: path to directory that contains all the h5 files for dataset
        :param trainsize: size of training set
        :param validsize: size of valid set, default = 500; to fix across train,valid,test retrieval
        :param testsize: size of test set, default = 2000; to fix across train,valid,test retrieval
        :param predict: DOS or bandstructures
        :param mode: ELdiff if predict difference from empty lattice, raw if predict the raw property
        :param split: to retrieve train, validation or test set
        :param dataset: blob or gpm

        """

        self.mode = mode
        self.input_size = 32

        if dataset == 'blob':
            filename = "sq-blob-v3-tm.h5"
            if targetmode == 'unlabeled':
                filename = "sq-blob-ul-15k-tm.h5"
            if split == 'test':
                filename = "sq-blob-test2k-tm.h5"

        elif dataset == 'gpm':
            filename = f"mf1-sg3-tm.h5"
            if targetmode == 'unlabeled':
                filename = f"mf1-sg3-25k-ul-s99-tm.h5"

        else:
            raise ValueError("Please specify dataset as blob or gpm.")

        totalstart = 1

        if split == 'train':
            indstart = totalstart
            indend = indstart + trainsize
        elif split == 'valid':
            indstart = totalstart + trainsize
            indend = indstart + validsize
        elif split == 'test':
            indstart = totalstart + trainsize + validsize
            indend = indstart + testsize

        indlist = range(indstart,indend)

        if ftsetind is not None:
            if split == 'train':
                indlist = ftsetind
            elif split == 'test':
                indlist = np.arange(totalstart + trainsize + validsize,indstart + testsize)

        if dataset == 'blob' and split == 'test':
            indlist = range(1,2000)

        print("loaded file: ", filename)
        self.len = len(indlist)

        ## initialize data lists
        self.x_data = []
        self.y_data = []
        self.EL_data = []
        self.ELd_data = []
        self.eps_data = []

        with h5py.File(os.path.join(path_to_h5_dir, filename), 'r') as f:
            for memb in indlist:
                inputeps = f['unitcell/mpbepsimage/'+str(memb)][()]
                epsavg = f["unitcell/epsavg/"+str(memb)][()]

                if predict == 'bandstructures':
                    y = f['mpbcal/efreq/'+str(memb)][()][:,:nbands]
                    el = f['mpbcal/emptylattice/'+str(memb)][()][:,:nbands]
                    eldiff = f['mpbcal/eldiff/'+str(memb)][()][:,:nbands]

                elif predict == 'DOS':
                    wvec = np.linspace(0,1.2,500)[ldos:udos]
                    if targetmode != 'unlabeled':
                        y = f['mpbcal/DOS/'+str(memb)][()][ldos:udos]/np.sqrt(epsavg)
                        el = wvec*2*math.pi
                        eldiff = y-el
                    else:
                        y = [0.]
                        eldiff = [0.]
                        el = [0.]
                else:
                    raise ValueError("Invalid property to predict. either DOS or bandstructures")

                self.x_data.append(inputeps)
                self.y_data.append(y)
                self.EL_data.append(el)
                self.ELd_data.append(eldiff)
                self.eps_data.append(epsavg)

        # normalize x data
        self.x_data = (np.array(self.x_data).astype('float32')-1) / 19 # normalize
        self.x_data = np.expand_dims(self.x_data,1) # add 1 channel for CNN
        self.y_data = np.array(self.y_data).astype('float32')
        self.EL_data = np.array(self.EL_data).astype('float32')
        self.ELd_data = np.array(self.ELd_data).astype('float32')
        self.eps_data = np.array(self.eps_data).astype('float32')


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        :return: random sample for a task
        """
        ## input always first element in tuple and output always second element
        if self.mode == 'raw':
            return self.x_data[index], self.y_data[index]
        elif self.mode == 'ELdiff':
            return self.x_data[index], self.ELd_data[index], self.y_data[index], self.EL_data[index]
