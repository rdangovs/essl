import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
import json

from utils import Cosine_Scheduler, DOSLoss
from ssl_mode import SimCLR, SimCLRwEE
from phc_models import Net
from phc_data import PhC2D
from symop_utils import get_aug, transform_and_get_labels
from load_config import update_args

def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def evaluate_model(args, dataloader, model, device):

    calL1loss = torch.nn.L1Loss()
    calDOSloss = DOSLoss()

    for step, task_set in enumerate(dataloader): # one step
        in_testdata = task_set[0].to(device)
        out_testdata = task_set[1].to(device)
        y_testdata = task_set[2].to(device)
        EL_testdata = task_set[3].to(device)

        output = model(in_testdata)
        ploss = calDOSloss(output+EL_testdata,y_testdata).item()

        return ploss


def pretrain_model(args, Encmodel, lr, criterion, pretrain_dataloader, device,
                    savestring, ptstring, eplist):

    print("We are pre-training using SSL mode {}".format(args.ssl_mode))

    opt = torch.optim.SGD(Encmodel.parameters(),momentum=0.9,lr=lr,weight_decay= 0)

    scheduler = Cosine_Scheduler(opt, num_epochs = max(eplist)+1, base_lr = lr, \
                                        iter_per_epoch = len(pretrain_dataloader))

    if args.log_to_tensorboard:
        tb_dir = 'runs_'+savestring +'/' + ptstring
        writer = SummaryWriter(tb_dir)

    parasum = 0
    for name, parameter in Encmodel.named_parameters():
        if not parameter.requires_grad: continue
        parasum+=parameter.numel()
    print("PT no. of Enc trainable params: ", parasum)

    for epoch in range(max(eplist)+1):
        if epoch in eplist:
            if not os.path.exists('./pretrained_models/'+savestring):
                os.makedirs('./pretrained_models/'+savestring)

            modeldict = {}
            modeldict.update(Encmodel.encoder.state_dict())
            torch.save({'model_state_dict': modeldict,
                        'ptepoch': epoch},
                        f'./pretrained_models/{savestring}/EP{epoch}{ptstring}')

        Encmodel.train()
        epochloss = 0.0
        epochloss1 = 0.0
        epochloss2 = 0.0

        ## Train backbone
        for step, load_set in enumerate(pretrain_dataloader):
            input_set = load_set[0]
            # get two views
            input1 = get_aug(input_set, translate=args.translate_pbc,\
            rotate=(args.rotate or args.flip_rotate),flip=(args.flip or args.flip_rotate),scale=args.scale, \
            translate_cont=args.trans_cont).to(device)
            input2 = get_aug(input_set, translate=args.translate_pbc,\
            rotate=(args.rotate or args.flip_rotate),flip=(args.flip or args.flip_rotate),scale=args.scale, \
            translate_cont=args.trans_cont).to(device)

            if args.ssl_mode == 'simclr_ee':
                rotinput, rotlabel = transform_and_get_labels(input_set, args.rotnet_trans, args.rotnet_flip, args.rotnet_rotate,nt=2)
                rotinput = rotinput.to(device)
                rotlabel = rotlabel.to(device)

            opt.zero_grad()

            if args.ssl_mode == 'simclr_ee':
                loss1, loss2 = Encmodel(input1,input2,rotinput,rotlabel)
                loss = loss1 + loss2 * args.alpha
            else:
                loss = Encmodel(input1,input2)
            loss.backward()
            epochloss += loss.item()

            if args.ssl_mode == 'simclr_ee':
                epochloss1 += loss1.item()
                epochloss2 += loss2.item()

            opt.step() # Update Enc and Dec
            scheduler.step()

        if args.log_to_tensorboard:
            writer.add_scalar('total_loss', epochloss/(step+1), epoch)
            if args.ssl_mode == 'simclr_ee':
                writer.add_scalar('sc_loss', epochloss1/(step+1), epoch)
                writer.add_scalar('ee_loss', epochloss2/(step+1), epoch)

        if args.ssl_mode == 'simclr' or args.ssl_mode == 'simclr_trans':
            epochloss1 = epochloss
        print(f"Epoch {epoch}: CL loss {epochloss1/(step+1)};  EE loss {epochloss2/(step+1)}")

    if args.log_to_tensorboard:
        writer.close()

def finetune_model(args, model, ftlr, criterion, train_dataloader, test_dataloader,
                    device, savestring, ftstring, PTEP, mintestloss=99.):

    opt = torch.optim.Adam(model.parameters(),lr=ftlr, weight_decay=1e-5)
    model_dir = './pretrained_models/'+savestring+'/'+"EP{}".format(PTEP)+ftstring

    if args.log_to_tensorboard:
        tb_dir = 'runs_'+savestring +'/' + ftstring+\
            "_EP{}_ftlr{}_ftseed{}_nsam{}_fe{}".format(PTEP,ftlr,args.ftseed,args.nsamargs,args.freeze_enc_ft)
        writer = SummaryWriter(tb_dir)

    model_dict = model.state_dict()
    checkpoint = torch.load(model_dir,map_location = torch.device(device))
    pretrained_dict = checkpoint['model_state_dict']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Loaded SSL state dict at PTEP ", PTEP)

    if args.freeze_enc_ft: ## freeze whole enc
        for name, param in model.named_parameters():
            if ("enc" in name):
                param.requires_grad = False
    parasum = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        parasum+=parameter.numel()
    print("Number of trainable params: ", parasum)

    for epoch in range(1,args.finetune_epochs+1):
        with torch.no_grad():
            model.eval() # evaluate model at every epoch
            ploss = evaluate_model(args, test_dataloader, model, device)
            if args.log_to_tensorboard:
                writer.add_scalar("FT_test_ploss", ploss, epoch)

            print(f"Test ploss at FT epoch {epoch}: {ploss}")
            mintestloss = np.min([mintestloss,ploss])

        model.train()
        epochloss = 0.

        for step, task_set in enumerate(train_dataloader): # batchsize of batchsize

            in_data = task_set[0]
            in_data = get_aug(in_data, translate=args.translate_pbc,\
            rotate=(args.rotate or args.flip_rotate),flip=(args.flip or args.flip_rotate),scale=args.scale, \
            translate_cont=args.trans_cont).to(device)

            out_data = task_set[1].to(device)

            opt.zero_grad()
            loss = criterion(model(in_data), out_data)
            loss.backward()
            epochloss += loss.item()
            opt.step()

        if args.log_to_tensorboard:
            writer.add_scalar("FT_train_loss", epochloss/(step+1), epoch)

    if args.log_to_tensorboard:
        writer.close()

    return mintestloss

def main(args):

    # load config
    args = update_args(args, dataset=args.dataset, ssl_mode=args.ssl_mode)
    # Define device and reset seed
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    reset_seeds(args.seed)

    # Create path for pretrained models
    if not os.path.exists('./pretrained_models'):
        os.makedirs('./pretrained_models')

    # Define network architecture and strings for logging
    ks = 7
    node = 256
    Lv = [64,256,256,1024,1024]
    Lvp = [1024,1024,512]
    Lvpj = [512,256]

    savestring = f'{args.dataset}_{args.ssl_mode}'
    ptstring = 'bscl{}_temp{}_ptlr{}'.format(args.batchsize_ssl,args.temperature,args.learning_rate)

    # Load datasets
    print("Retrieving data.. ")
    totalsam = 6000
    if args.dataset == 'gpm':
        totalsam = 9800
        trainsam = totalsam - 2000
    elif args.dataset == 'blob':
        trainsam = 5100

    pick_random_ftset = np.random.choice(np.arange(1,trainsam),args.nsam,replace=False)
    test_dataloader = None

    if not args.no_finetune:
        traindataset = PhC2D(args.path_to_h5, args.nsam, validsize = 0, testsize = 2000,
                                split = 'train', ftsetind = pick_random_ftset,
                                dataset=args.dataset)
        testdataset = PhC2D(args.path_to_h5, args.nsam , validsize = trainsam-args.nsam,
                                testsize = 2000, split = 'test', ftsetind = pick_random_ftset,
                                dataset=args.dataset)

        train_dataloader = data.DataLoader(traindataset, batch_size = args.batchsize,shuffle = True)
        test_dataloader = data.DataLoader(testdataset, batch_size=len(testdataset))
        print("Finetuning data retrieved!")

    ## Define Loss criterion
    ft_criterion = torch.nn.MSELoss()
    pt_criterion = torch.nn.MSELoss()

    ## Training modules
    if not args.no_pretrain:
    # Load new unlabeled datasets for pretext_surr model training
        ntarget = 20480 if args.dataset == 'gpm' else 15000 # blob dataset is small

        print("Retrieving unlabelled target data set..")
        pretrain_dataset = PhC2D(args.path_to_h5, ntarget, validsize = 0, testsize =0,
                                targetmode = 'unlabeled',
                                split = 'train', dataset=args.dataset)

        pretrain_dataloader = data.DataLoader(pretrain_dataset,batch_size = args.batchsize_ssl,drop_last = True)
        print("Unlabeled data retrieved!")

    eplist = [100,200] # epochs to save model

    if not args.no_pretrain:
        if args.ssl_mode == 'simclr' or args.ssl_mode == 'simclr_trans':
            Encnet = SimCLR(Lv,Lvpj,ks,device,args.batchsize_ssl,
            args.temperature,use_projector=True).to(device)
        elif args.ssl_mode == 'simclr_ee':
            Encnet = SimCLRwEE(Lv,Lvpj,ks,device,args.batchsize_ssl,
                    args.temperature,use_projector=True,
                    nrot=2**2).to(device)
            print("alpha = ",args.alpha)

        pretrain_model(args, Encnet, args.learning_rate,pt_criterion, \
                pretrain_dataloader, device, savestring, ptstring, eplist)

    mintestloss = 99.

    if not args.no_finetune:
        reset_seeds(args.ftseed)
        for PTEP in eplist:
            model = Net(Lv,ks,Lvp).to(device)
            # optimizer will be initialized in module
            mintestloss = finetune_model(args,model,args.learning_rate_ft, ft_criterion,\
            train_dataloader, test_dataloader, device, savestring, ptstring, PTEP, mintestloss)

    return mintestloss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Data and task parameters
    parser.add_argument('--path_to_h5', type = str, default = './datasets')
    parser.add_argument('--device',type=int, help="Device number. Default = 0 ", default = 0)
    parser.add_argument('--log_to_tensorboard', action='store_true', help = 'if logging to tensorboard')
    parser.add_argument('--dataset', type = str, default = 'blob', choices=['blob', 'gpm'])
    parser.add_argument('--nsam', type=int, help='no. of labelled target training samples',default=3000)

    # Training parameters
    parser.add_argument('--batchsize', type=int, help='batchsize for finetuning. Default = 64',default=64)
    parser.add_argument('--learning_rate',type=float, help='pretraining learning rate', default=1e-3)
    parser.add_argument('--learning_rate_ft',type=float, help='finetuning learning rate', default=1e-4)
    parser.add_argument('--finetune_epochs',type=int, help='total no. of epochs to finetune', default=100)
    parser.add_argument('--seed',type=int, help='Random seed', default=1)
    parser.add_argument('--ftseed',type=int, help='Random ft seed', default=1)
    parser.add_argument('--freeze_enc_ft',action='store_true',help="Set this flag to freeze enc during finetuning")

    # SSL specific parameters
    parser.add_argument('--ssl_mode', type = str, default = 'simclr', choices=['simclr','simclr_ee','simclr_trans'])
    parser.add_argument('--batchsize_ssl', type=int, help='specify batchsize for SSL', default=256)
    parser.add_argument('--temperature',type=float, help='specify temperature for CL loss function', default=0.1)
    parser.add_argument('--rotnet_trans', action='store_true')
    parser.add_argument('--rotnet_flip', action='store_true')
    parser.add_argument('--rotnet_rotate', action='store_true')
    parser.add_argument('--translate_pbc', action='store_true', help = 'to randomly translate input image or not (to take care of PBC)')
    parser.add_argument('--flip_rotate', action='store_true', help = 'to randomly flip and then rotate image')
    parser.add_argument('--flip', action='store_true', help = 'to randomly flip image')
    parser.add_argument('--rotate', action='store_true', help = 'to randomly rotate image')
    parser.add_argument('--scale', action='store_true', help = 'to randomly scale input by factor in (0,1]')

    parser.add_argument('--alpha',type=float, default=0.0)
    parser.add_argument('--trans_cont', action='store_true')

    parser.add_argument('--no_pretrain',action='store_true',help="Set this flag to not pretrain")
    parser.add_argument('--no_finetune',action='store_true',help="Set this flag to not finetune")

    args = parser.parse_args()

    minloss = main(args)
    losses = [minloss]
    for ftseed in [2,3]:
        args.no_pretrain = True
        args.ftseed = ftseed
        minloss = main(args)
        losses.append(minloss)

    print(f"Average loss over 3 seeds: {np.mean(losses)} +/- {np.std(losses)}")
