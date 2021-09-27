import glob
import math
import tqdm
import torch
import random
import inspect
import os.path as osp
import torch.nn as nn
from pathlib import Path
from itertools import chain
from torch.utils.data import random_split
from torch_geometric.data import Data, DataLoader, DataListLoader
from torch_geometric.nn import EdgeConv, global_mean_pool, DataParallel

import models.models as models
import models.emd_models as emd_models
from util.loss_util import LossFunction
from datagen.graph_data_gae import GraphDataset
from util.train_util import get_model, forward_loss
from util.preprocessing import get_iqr_proportions, standardize
from util.plot_util import loss_curves, epoch_emd_corr, plot_reco_for_loader, plot_emd_corr

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
multi_gpu = torch.cuda.device_count()>1

@torch.no_grad()
def test(model, loader, total, batch_size, loss_ftn_obj, gen_emd_corr=False, scaler=None):
    model.eval()

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    if gen_emd_corr:
        in_parts = []
        gen_parts = []
        pred_emd = []

    for i,data in t:

        batch_loss, batch_output = forward_loss(model, data, loss_ftn_obj, device, multi_gpu, scaler)
        if 'emd_loss' in loss_ftn_obj.name:
            batch_loss, true_emd = batch_loss

        batch_loss = batch_loss.item()
        sum_loss += batch_loss
        t.set_description('eval loss = %.7f' % (batch_loss))
        t.refresh() # to show immediately the update

        if gen_emd_corr:
            in_parts.append(data.x.detach().cpu().numpy())
            gen_parts.append(batch_output.detach().cpu().numpy())
            pred_emd.append(batch_loss)

    if gen_emd_corr:
        return sum_loss / (i+1), in_parts, gen_parts, pred_emd
    if 'emd_loss' in loss_ftn_obj.name:
        return sum_loss / (i+1), true_emd
    return sum_loss / (i+1)

def train(model, optimizer, loader, total, batch_size, loss_ftn_obj, scaler=None):
    model.train()

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        optimizer.zero_grad()

        batch_loss, batch_output = forward_loss(model, data, loss_ftn_obj, device, multi_gpu, scaler)
        if 'emd_loss' in loss_ftn_obj.name:
            batch_loss, true_emd = batch_loss
        batch_loss.backward()
        optimizer.step()

        batch_loss = batch_loss.item()
        sum_loss += batch_loss
        t.set_description('train loss = %.7f' % batch_loss)
        t.refresh() # to show immediately the update

    if 'emd_loss' in loss_ftn_obj.name:
        return sum_loss / (i+1), true_emd
    return sum_loss / (i+1)

def main(args):
    model_fname = args.mod_name

    if multi_gpu and args.batch_size < torch.cuda.device_count():
        exit('Batch size too small')
    if args.loss == 'deepemd_loss' and args.batch_size > 1:
        exit('deepemd_loss can only be used with batch_size of 1 for now')

    # make a folder for the graphs of this model
    Path(args.output_dir).mkdir(exist_ok=True)
    save_dir = osp.join(args.output_dir, model_fname)
    Path(save_dir).mkdir(exist_ok=True)

    # dataset
    gdata = GraphDataset(root=args.input_dir, bb=args.box_num)
    # merge data from separate files into one contiguous array
    dataset = [data for data in chain.from_iterable(gdata)]
    random.Random(0).shuffle(dataset)
    dataset = dataset[:args.num_data]

    fulllen = len(dataset)
    train_len = int(0.8 * fulllen)
    tv_len = int(0.10 * fulllen)
    train_dataset = dataset[:train_len]
    valid_dataset = dataset[train_len:train_len + tv_len]
    test_dataset  = dataset[train_len + tv_len:]
    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)
    test_samples = len(test_dataset)
    num_workers = args.num_workers
    if multi_gpu:
        train_loader = DataListLoader(train_dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
        valid_loader = DataListLoader(valid_dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
        test_loader  = DataListLoader(test_dataset,  batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    # preprocessing options
    iqr_prop = None
    if 'iqr' in args.loss:
        iqr_prop = get_iqr_proportions(train_dataset)
    scaler = None
    if args.standardize:
        scaler = standardize(train_dataset, valid_dataset, test_dataset)

    loss_ftn_obj = LossFunction(args.loss, emd_model_name=args.emd_model_name, device=device, iqr_prop=iqr_prop)

    # model
    input_dim = 3
    big_dim = 32
    hidden_dim = args.lat_dim
    model = get_model(args.model, input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim, emd_modname=args.emd_model_name)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, threshold=1e-6)

    # load model
    valid_losses = []
    train_losses = []
    train_true_emd = [] if 'emd_loss' in loss_ftn_obj.name else None
    valid_true_emd = [] if 'emd_loss' in loss_ftn_obj.name else None
    start_epoch = 0
    modpath = osp.join(save_dir, model_fname+'.best.pth')
    if osp.isfile(modpath):
        model.load_state_dict(torch.load(modpath, map_location=device))
        model.to(device)
        if not args.drop_old_losses:    # use when swapping from pretrained network to new training w/ different loss
            best_valid_loss = test(model, valid_loader, valid_samples, args.batch_size, loss_ftn_obj, scaler=scaler)
            if 'emd_loss' in loss_ftn_obj.name:
                best_valid_loss, ef_emd = best_valid_loss
            print('Loaded model')
            print(f'Saved model valid loss: {best_valid_loss}')
            if osp.isfile(osp.join(save_dir,'losses.pt')):
                train_losses, valid_losses, start_epoch = torch.load(osp.join(save_dir,'losses.pt'))
        else:
            best_valid_loss = 9999999
    else:
        print('Creating new model')
        best_valid_loss = 9999999
        model.to(device)
    if multi_gpu:
        model = DataParallel(model)
        model.to(device)

    # Training loop
    n_epochs = 200
    stale_epochs = 0
    loss = best_valid_loss
    for epoch in range(start_epoch, n_epochs):

        loss = train(model, optimizer, train_loader, train_samples, args.batch_size, loss_ftn_obj, scaler=scaler)
        if 'emd_loss' in loss_ftn_obj.name:
            loss, ef_emd = loss
            train_true_emd.append(ef_emd)
        # if epoch % 5 == 0 and args.loss == 'emd_loss':
        #     valid_loss, in_parts, gen_parts, pred_emd = test(model, valid_loader, valid_samples, args.batch_size, loss_ftn_obj, True, scaler=scaler)
        #     epoch_emd_corr(in_parts, gen_parts, pred_emd, save_dir, epoch)
        # else:
        valid_loss = test(model, valid_loader, valid_samples, args.batch_size, loss_ftn_obj, scaler=scaler)
        if 'emd_loss' in loss_ftn_obj.name:
            valid_loss, ef_emd = valid_loss
            valid_true_emd.append(ef_emd)

        scheduler.step(valid_loss)
        train_losses.append(loss)
        valid_losses.append(valid_loss)
        print('Epoch: {:02d}, Training Loss:   {:.4f}'.format(epoch, loss))
        print('               Validation Loss: {:.4f}'.format(valid_loss))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('New best model saved to:',modpath)
            if multi_gpu:
                torch.save(model.module.state_dict(), modpath)
            else:
                torch.save(model.state_dict(), modpath)
            torch.save((train_losses, valid_losses, epoch+1), osp.join(save_dir,'losses.pt'))
            stale_epochs = 0
        else:
            stale_epochs += 1
            print(f'Stale epoch: {stale_epochs}\nBest: {best_valid_loss}\nCurr: {valid_loss}')
        if stale_epochs >= args.patience:
            print('Early stopping after %i stale epochs'%args.patience)
            break

    # model training done
    train_epochs = list(range(epoch+1))
    early_stop_epoch = epoch - stale_epochs
    loss_curves(train_epochs, early_stop_epoch, train_losses, valid_losses, save_dir, train_true_emd, valid_true_emd)

    # load best model
    del model
    torch.cuda.empty_cache()
    model = get_model(args.model, input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim, emd_modname=args.emd_model_name)
    model.load_state_dict(torch.load(modpath))
    if multi_gpu:
        model = DataParallel(model)
    model.to(device)

    inverse_standardization = args.standardize and args.plot_scale != 'standardized'
    plot_reco_for_loader(model, train_loader, device, scaler, inverse_standardization, model_fname, osp.join(save_dir, 'reconstruction_post_train', 'train'), args.plot_scale)
    plot_reco_for_loader(model, valid_loader, device, scaler, inverse_standardization, model_fname, osp.join(save_dir, 'reconstruction_post_train', 'valid'), args.plot_scale)
    plot_reco_for_loader(model, test_loader, device, scaler, inverse_standardization, model_fname, osp.join(save_dir, 'reconstruction_post_train', 'test'), args.plot_scale)
    
    # plot emd correlation plots between emd-nn and true emd values between gae input and output
    if args.plot_emd_corr:
        loss_ftn_obj = LossFunction('emd_loss', emd_model_name=args.emd_model_name, device=device)
        emd_loss_ftn = loss_ftn_obj.loss_ftn
        plot_emd_corr(model, test_loader, emd_loss_ftn, save_dir, 'emd_corr', scaler, device)
    print('Completed')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod-name', type=str, help='model name for saving and loading', required=True)
    parser.add_argument('--input-dir', type=str, help='location of dataset', required=True)
    parser.add_argument('--output-dir', type=str, help='root folder to output experiment results to', 
                        default='/anomalyvol/experiments/', required=False)
    parser.add_argument('--box-num', type=int, help='0=QCD-background; 1=bb1; 2=bb2; 4=rnd', default=0, required=False)
    parser.add_argument('--lat-dim', type=int, help='latent space size', default=2, required=False)
    parser.add_argument('--model', 
                        choices=[m[0] for m in inspect.getmembers(models, inspect.isclass) if m[1].__module__ == 'models.models'], 
                        help='model selection', required=True)
    parser.add_argument('--batch-size', type=int, help='batch size', default=2, required=False)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3, required=False)
    parser.add_argument('--patience', type=int, help='patience', default=10, required=False)
    parser.add_argument('--loss', choices=[m for m in dir(LossFunction) if not m.startswith('__')], 
                        help='loss function', required=True)
    parser.add_argument('--emd-model-name', choices=[osp.basename(x).split('.')[0] for x in glob.glob('/anomalyvol/emd_models/*')], 
                        help='emd models for loss', default='EmdNNSpl', required=False)
    parser.add_argument('--num-data', type=int, help='how much data to use (e.g. 10 jets)', 
                        default=None, required=False)
    parser.add_argument('--num-workers', type=int, help='num_workers param for dataloader', 
                        default=0, required=False)
    parser.add_argument("--standardize", action="store_true", help="normalize dataset", required=False)
    parser.add_argument("--plot-emd-corr", action="store_true", help="plot emd correlation plot at end of training", required=False)
    parser.add_argument("--drop-old-losses", action="store_true", help="don't load in old loss values", required=False)
    parser.add_argument("--plot-scale", choices=['cartesian','hadronic','standardized'],
                        help='classes for x-axis scaling for plotting reconstructions', default='cartesian', required=False)
    args = parser.parse_args()

    main(args)
