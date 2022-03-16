
def update_args(args, dataset = 'blob', ssl_mode = 'simclr'):
    args.nsam = 3000
    args.alpha = 5.

    if dataset == 'blob':
        args.flip_rotate = True
        if ssl_mode == 'simclr_ee':
            args.rotnet_trans = True
        elif ssl_mode == 'simclr_trans':
            args.translate_pbc = True

    elif dataset == 'gpm':
        args.translate_pbc = True
        args.trans_cont = True

        args.flip = True
        if ssl_mode == 'simclr_ee':
            args.rotnet_rotate = True

        elif ssl_mode == 'simclr_trans':
            args.rotate = True

    return args


'''
BLOB:
python main_phc.py --dataset=blob --ssl_mode=simclr
python main_phc.py --dataset=blob --ssl_mode=simclr_ee
python main_phc.py --dataset=blob --ssl_mode=simclr_trans 
GPM:
python main_phc.py --dataset=gpm --ssl_mode=simclr
python main_phc.py --dataset=gpm --ssl_mode=simclr_ee
python main_phc.py --dataset=gpm --ssl_mode=simclr_trans


'''
