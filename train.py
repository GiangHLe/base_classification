import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse
import pandas as pd

from code.datasets import get_dataLoader
from code.utils import *
from code.models import *
from code.schedule_lr import SnapShotLearningRate

import torch
from torch import nn as nn
from tqdm import tqdm

def run(save_path, dataloader, epochs, warm_up_epoch, optimizer, lr, num_cycle, weight_decay, weight_path, log_cache, num_sample_train, device):
    '''
    Train model with warm up new layer and RandAugment, this function will:
        + Write three log file: 
            + model architecture
            + training information (num batchsize, optimizer, learning rate, criterion/loss function)
            + Loss value, accuracy or any evaluation result per epochs
        + Dump path file contain:
            + Optimizer parameter: to make sure it will be similar (momentum, or others accelaration vector) if we want to continue on training
            + Model parameters
        + A json file which contain the information to draw the graph.
        + TODO: Adding snapshot
    '''
    train_loader, val_loader, warm_loader = dataloader
    # nn.CrossEntropyLoss will auto input the ouput with softmax, we need to solve multi-class classification so we need sigmoid here 
    criterion = SigmoidCrossEntropy()
    # Select model, model should have freeze function
    model = ProblemModel()
    # get optimizer
    optimizer = get_optimizer(optimizer, model, lr, weight_decay)

    last_epoch = 0
    if weight_path:
        last_epoch = extract_number(weight_path)
        try:
            checkpoint = torch.load(weight_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            if checkpoint['pre_opt'] == args.opt:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            model.load_state_dict(torch.load(args.weights))
    
    model.to(device)

    write_log(
        save_path, 
        model, 
        optimizer, 
        criterion,
        log_cache
    )

    plot_dict = {'train': list(), 'val': list()}
    
    log_train_path = save_path + 'training_log.txt'
    plot_train_path = save_path + 'log.json'

    write_mode = 'w'

    ### edit if you need to plot more
    if os.path.exists(log_train_path) and os.path.exists(plot_train_path):
        write_mode = 'a'
        with open(plot_train_path, 'r') as j:
            plot_dict = json.load(j)
            plot_dict['train'] = plot_dict['train'][:last_epoch]
            plot_dict['val'] = plot_dict['val'][:last_epoch]
    else:
        # Training with warm up term
        print('Start warm up')
        model.freeze_old_layer()
        for epoch in range(warm_up_epoch):
            warm_up(model= model,
                    dataloader= warm_loader, 
                    optimizer = optimizer, 
                    criterion = criterion,
                    device = device,
                    )
        model.unfreeze()
    print('Start training')

    if args.nc:
        step_per_snap = int(epochs * (num_sample_train//args.bs) / num_cycle)
        scheduler = SnapShotLearningRate(
            optimizer = optimizer,
            based_lr = lr,
            step_per_snap = step_per_snap,
        )
    else:
        scheduler = optimizer
        step_per_snap = 1

    num_step = 0

    with open(log_train_path, write_mode) as f:
        for epoch in range(1, epochs+1):
            real_epoch = epoch + last_epoch
            print('Epoch:',real_epoch)
            f.write('Epoch: %d\n'%(real_epoch))
            loss_train, num_step = train_epoch(model = model,
                               dataloader = train_loader, 
                               optimizer = scheduler, # apply snapshot for training
                               criterion = criterion,
                               evaluation = evaluation,
                               step_per_snap = step_per_snap, 
                               num_step = num_step,
                               save_path = args.save_path,
                               device = device
                            )
            loss_val = val_epoch(model = model,
                             dataloader=val_loader,
                             criterion = criterion,
                             evaluation = evaluation, 
                             device=device
                            )
            ### should depend on the evaluation metric that they required
            loss_train, _ = loss
            loss_val, _ = loss_val
            f.write('Training loss: %.4f\n'%(loss_train))
            f.write('Validation loss: %.4f\n'%(loss_val))
            f.write('-'*20)
            print('Training loss: %.4f'%(loss_train))
            print('Validation loss: %.4f'%(loss_val))

            # torch.save(model.state_dict(), save_path + 'epoch%d.pth'%(epoch+last_epoch))
            save_name = save_path + 'epoch%d.pth'%(real_epoch)
            save_pth(save_name , real_epoch, model, optimizer, args.opt)

            # edit here if have more thing want to plot
            plot_dict['train'].append(loss_train) 
            plot_dict['val'].append(loss_val)
            with open(plot_train_path, 'w') as j:
                json.dump(plot_dict, j)

def train():
    '''
    Train function, in case we need K-Fold training
    '''
    weight_path = args.w 
    num_cycle = 0
    if args.nc:
        num_cycle = args.nc
    if args.fold:
        df = grab_df(args.dp)
        order = True
    else:
        df = [pd.read_csv(args.dp)]
        order = False
    for k, dataf in enumerate(df):
        train_loader, val_loader, warm_loader, log_cache, n_train = get_dataLoader(
            dataframe = dataf,
            image_size = args.is,
            batch_size = args.bs,
            num_workers = args.nw,
            N = args.N,
            M = args.M,
            shuffle = args.no_shuffle
        )
        save_path = args.save_path
        if order:
            save_path = save_path[:-1] + '_fold%d/'%(k) # in case k-fold
        check_path(save_path) # check whether it have the dir to save or not
        run(
            save_path = save_path,
            dataloader = (train_loader, val_loader, warm_loader), 
            epoch = args.epoch,
            warm_up_epoch = args.wue,
            optimizer = args.opt,
            lr = args.lr,
            num_cycle = num_cycle,
            weight_decay = args.wd,
            weight_path = weight_path,
            log_cache = log_cache,
            num_sample_train = n_train,
            device = device
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
     
    parser.add_argument('save_path', type = str, help = 'Directory where to save model and plot information')
    parser.add_argument('-w', '--weights', type = str, default = None, help = 'Weights path')
    parser.add_argument('-dp', '--data_path', type = str, default='./data/', help = 'where to store csv file')
    parser.add_argument('-e', '--epoch', type = int, default=200, help = 'Number of epochs')
    parser.add_argument('-bs', '--batch-size', type = int, default=15, help = 'Batch size')
    parser.add_argument('-lr', '--learning-rate', type = float, default=3e-4, help = 'Learning rate')
    parser.add_argument('-opt', '--optimize', type = str, default='radam', help = 'Select optimizer')
    parser.add_argument('-is', '--image-size', type = int, default=366, help = 'Size of image input to models')
    parser.add_argument('-n', type = int, default= 3, help = 'Number of transformations we will -> RandAugment')
    parser.add_argument('-m', type = int, default=10, help = 'the scale for strength of transformations -> RandAugment')
    parser.add_argument('-nw', '--num-workers', type = int, default=8, help = 'Number of process in Datat Loader')
    parser.add_argument('-wd', '--weight-decay', type = float, default=0.01, help = 'L2 Regularization')
    parser.add_argument('-wue', '--warm-up', type = int, default=3, help = 'Number of warm up epochs')
    parser.add_argument('-nc','--num-cycle', type = int, default=None, help = 'Number of intermediate models if apply snapshot')
    parser.add_argument('--fold', type = int, default=None, help = 'Number of fold')
    parser.add_argument('--no-shuffle', action='store_true', help = 'shuffle while training')
    
    args = parser.parse_args()

    train()
