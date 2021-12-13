import glob
import tqdm
import torch
import random
import inspect
import numpy as np
import os.path as osp
from pathlib import Path
from itertools import chain
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel

import models.models as models
import models.emd_models as emd_models
from util.loss_util import LossFunction
from datagen.graph_data_gae import GraphDataset
from util.train_util import get_model, forward_loss
from util.preprocessing import get_iqr_proportions, standardize
from util.plot_util import loss_curves, plot_reco_for_loader, plot_emd_corr, adv_loss_curves, epoch_emd_corr
from util.adversarial import loop_emd_model

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
multi_gpu = torch.cuda.device_count()>1

@torch.no_grad()
def test(model, loader, total, batch_size, loss_ftn_obj, scaler=None, gen_emd_corr=False, emd_adv_train=False, emd_optimizer=None, emd_adv_patience=4, fig_save_path=None, emd_model_path=None):
    model.eval()

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    if gen_emd_corr:
        in_parts = []
        gen_parts = []
        pred_emd = []

    for i,data in t:

        # overfit the emd network on this batch
        if emd_adv_train:
            max_iters = 20
            batch_num = i
            loss_ftn_obj.emd_model.load_state_dict(torch.load(emd_model_path, map_location=device))
            _, _, _ = loop_emd_model(model, loss_ftn_obj.emd_model, emd_optimizer, data, scaler, max_iters, emd_adv_patience, batch_num, fig_save_path, device)

        batch_loss, batch_output = forward_loss(model, data, loss_ftn_obj, device, multi_gpu, scaler)
        if 'emd_loss' in loss_ftn_obj.name:
            batch_loss, true_emd = batch_loss

        batch_loss = batch_loss.item()
        sum_loss += batch_loss
        t.set_description('eval loss = %.7f' % (batch_loss))
        t.refresh() # to show immediately the update

        if gen_emd_corr:
            in_parts.append(data.x.detach().cpu().numpy())
            if not isinstance(batch_output, np.ndarray):
                batch_output = batch_output.detach().cpu().numpy()
            gen_parts.append(batch_output)
            pred_emd.append(batch_loss)

    avg_loss = sum_loss / (i+1)
    if gen_emd_corr:
        return avg_loss, true_emd, in_parts, gen_parts, pred_emd
    elif 'emd_loss' in loss_ftn_obj.name:
        return avg_loss, true_emd
    return avg_loss

def train(model, optimizer, loader, total, batch_size, loss_ftn_obj, emd_optimizer, scaler=None, emd_adv_train=False, emd_adv_patience=4, fig_save_path=None, emd_model_path=None):
    model.train()

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i, data in t:
        optimizer.zero_grad()

        # overfit the emd network on this batch
        if emd_adv_train:
            max_iters = 20
            batch_num = i
            loss_ftn_obj.emd_model.load_state_dict(torch.load(emd_model_path, map_location=device))
            _, _, _ = loop_emd_model(model, loss_ftn_obj.emd_model, emd_optimizer, data, scaler, max_iters, emd_adv_patience, batch_num, fig_save_path, device)

        # train the gae
        batch_loss, _ = forward_loss(model, data, loss_ftn_obj, device, multi_gpu, scaler)

        if 'emd_loss' in loss_ftn_obj.name:
            batch_loss, true_emd = batch_loss
            # if emd_adv_train:   # early stop training if loss diverges too much from true emd
                # if (batch_loss - true_emd) / true_emd > 0.25:
                #     emd_adv_stale_epochs += 1
                #     if emd_adv_stale_epochs >= emd_adv_patience:
                #         i -= 1
                #         break

        batch_loss = batch_loss.item()
        sum_loss += batch_loss
        t.set_description('train loss = %.7f' % batch_loss)
        t.refresh() # to show immediately the update

    avg_loss = sum_loss / (i+1)
    if 'emd_loss' in loss_ftn_obj.name:
        return avg_loss, true_emd

    return avg_loss

def main(args):
    model_fname = args.mod_name

    if multi_gpu and args.batch_size < torch.cuda.device_count():
        exit('Batch size too small')
    if args.loss == 'deepemd_loss' and args.batch_size > 1:
        exit('deepemd_loss can only be used with batch_size of 1 for now')
    if args.train_emd_adversarially and args.loss != 'emd_loss':
        exit('Must use emd_loss to train emd network adversarially')

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

    # loss function
    loss_ftn_obj = LossFunction(args.loss, emd_model_name=args.emd_model_name, device=device, iqr_prop=iqr_prop)
    if args.train_emd_adversarially:    # set up parameters for training emd nn with gae
        emd_model = loss_ftn_obj.emd_model
        emd_optimizer = torch.optim.Adam(emd_model.parameters(), lr = args.lr)
    else:
        emd_optimizer = None

    # model
    input_dim = 3
    big_dim = 32
    hidden_dim = args.lat_dim
    model = get_model(args.model, input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim, emd_modname=args.emd_model_name)

    valid_losses = []
    train_losses = []
    gae_train_true_emd = [] if 'emd_loss' in loss_ftn_obj.name else None    # energyflow emd with gae reco for train epochs
    gae_valid_true_emd = [] if 'emd_loss' in loss_ftn_obj.name else None
    train_adv_loss = [] if args.train_emd_adversarially else None
    valid_adv_loss = [] if args.train_emd_adversarially else None
    train_emd_pred = [] if args.train_emd_adversarially else None
    valid_emd_pred = [] if args.train_emd_adversarially else None
    emdnn_train_emd_true = [] if args.train_emd_adversarially else None
    emdnn_valid_emd_true = [] if args.train_emd_adversarially else None
    start_epoch = 0
    lr = args.lr
    best_valid_loss = 9999999   # default valid loss
    modpath = osp.join(save_dir, model_fname+'.best.pth')
    emdmodpath = osp.join(save_dir, 'emd_model.pt')

    # load model and previous training state
    if osp.isfile(modpath):
        model.load_state_dict(torch.load(modpath, map_location=device))
        model.to(device)

        if osp.isfile(emdmodpath) and args.train_emd_adversarially: # load emd-nn
            emd_model.load_state_dict(torch.load(emdmodpath, map_location=device))

        # calculate best validation loss
        best_valid_loss = test(model, valid_loader, valid_samples, args.batch_size, loss_ftn_obj, scaler=scaler)
        if 'emd_loss' in loss_ftn_obj.name:
            best_valid_loss, ef_emd = best_valid_loss
            if args.train_emd_adversarially:
                best_valid_loss = ef_emd
        print(f'Loaded Model\nSaved model valid loss: {best_valid_loss}')

        if not args.drop_old_losses:    # load previous train info (losses, etc.)
            if osp.isfile(osp.join(save_dir, 'train_status.pt')):
                train_status = torch.load(osp.join(save_dir, 'train_status.pt'))
                train_losses = train_status['train_losses']
                valid_losses = train_status['valid_losses']
                start_epoch = train_status['epoch']
                lr = train_status['lr']
                gae_train_true_emd = train_status['gae_train_true_emd']
                gae_valid_true_emd = train_status['gae_valid_true_emd']
                emdnn_train_emd_true = train_status['emdnn_train_emd_true']
                emdnn_valid_emd_true = train_status['emdnn_valid_emd_true']
    else:
        print('Creating new model')
        model.to(device)
    if multi_gpu:
        model = DataParallel(model)
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, threshold=1e-6)

    # Training loop
    n_epochs = args.epochs
    stale_epochs = 0

    for epoch in range(start_epoch, n_epochs):
        print("=" * 50)
        print('Epoch: {:02d}'.format(epoch))

        # # Adv. EMD-NN train loop
        # if args.train_emd_adversarially:
        #     best_emd_valid_loss = 9999999
        #     emd_stale_epochs = 0
        #     while True:
        #         # train/validate
        #         emd_train_loss, emd_pred_avg_train, emd_true_avg_train = loop_emd_model(model, emd_model, emd_optimizer, train_loader, scaler, device)
        #         emd_valid_loss, emd_pred_avg_valid, emd_true_avg_valid = loop_emd_model(model, emd_model, None, train_loader, scaler, device)

        #         # save values for plotting
        #         train_adv_loss.append(emd_train_loss)
        #         valid_adv_loss.append(emd_valid_loss)
        #         train_emd_pred.append(emd_pred_avg_train)
        #         valid_emd_pred.append(emd_pred_avg_valid)
        #         emdnn_train_emd_true.append(emd_true_avg_train)
        #         emdnn_valid_emd_true.append(emd_true_avg_valid)
        #         print('EMD-NN Training Loss: {:.4f}'.format(emd_train_loss))
        #         print('EMD-NN Valid Loss: {:.4f}'.format(emd_valid_loss))

        #         # early stopping
        #         if emd_valid_loss < best_emd_valid_loss:
        #             best_emd_valid_loss = emd_valid_loss
        #             emd_stale_epochs = 0
        #             torch.save(emd_model.state_dict(), emdmodpath)
        #         else:
        #             emd_stale_epochs += 1
        #             if emd_stale_epochs >= args.emd_nn_patience:
        #                 emd_model.load_state_dict(torch.load(emdmodpath, map_location=device))
        #                 if args.plot_emd_corr_epoch != 0 and epoch % args.plot_emd_corr_epoch == 0:
        #                     emd_loss_ftn = loss_ftn_obj.loss_ftn
        #                     # plot_emd_corr(model, valid_loader, emd_loss_ftn, save_dir, f'emd_corr_valid_{epoch}', scaler, device, sub_dir='post_emd_loop_corr')
        #                 break

        # train GAE
        loss = train(model,
                     optimizer,
                     train_loader,
                     train_samples,
                     args.batch_size,
                     loss_ftn_obj,
                     emd_optimizer,
                     scaler=scaler,
                     emd_adv_train=args.train_emd_adversarially,
                     fig_save_path=osp.join(save_dir,'emd_train_curves'),
                     emd_model_path=emdmodpath)
        if 'emd_loss' in loss_ftn_obj.name:     # for emd_loss unpack emd target from emd pred
            train_loss, ef_emd = loss
            gae_train_true_emd.append(ef_emd)

        # validate GAE
        if args.plot_emd_corr_epoch != 0 and epoch % args.plot_emd_corr_epoch == 0:     # make emd corr plot at this epoch
            valid_loss, ef_emd, in_parts, gen_parts, pred_emd = test(model,
                                                                     valid_loader,
                                                                     valid_samples,
                                                                     args.batch_size,
                                                                     loss_ftn_obj,
                                                                     scaler=scaler,
                                                                     gen_emd_corr=True,
                                                                     emd_adv_train=False,
                                                                     emd_optimizer=None,
                                                                     emd_adv_patience=4,
                                                                     fig_save_path=None,
                                                                     emd_model_path=None)
            gae_valid_true_emd.append(ef_emd)
            sub_dir = "valid_gae_emd_corr"
            epoch_emd_corr(in_parts, gen_parts, pred_emd, save_dir, sub_dir, epoch)
        else:   # default
            valid_loss = test(model,
                              valid_loader,
                              valid_samples,
                              args.batch_size,
                              loss_ftn_obj,
                              scaler=scaler,
                              emd_adv_train=True,
                              emd_optimizer=emd_optimizer,
                              emd_adv_patience=4,
                              fig_save_path='emd_train_curve_gae_valid',
                              emd_model_path=emdmodpath)
            if 'emd_loss' in loss_ftn_obj.name:
                valid_loss, ef_emd = valid_loss
                gae_valid_true_emd.append(ef_emd)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print('Training Loss: {:.4f}'.format(train_loss))
        print('Validation Loss: {:.4f}'.format(valid_loss))

        if args.train_emd_adversarially:    # use true emd if training adv
            valid_loss = ef_emd
            print('Validation EMD True: {:.4f}'.format(ef_emd))

        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:    # early stop check
            stale_epochs = 0
            best_valid_loss = valid_loss

            print('New best model saved to:', modpath)
            if multi_gpu:
                torch.save(model.module.state_dict(), modpath)
            else:
                torch.save(model.state_dict(), modpath)

            train_status = {
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'epoch': epoch+1,
                'lr': optimizer.param_groups[0]['lr'],
                'gae_train_true_emd': gae_train_true_emd,
                'gae_valid_true_emd': gae_valid_true_emd
            }
            torch.save(train_status, osp.join(save_dir, 'train_status.pt'))
        else:
            stale_epochs += 1
            print(f'Stale epoch: {stale_epochs}\nBest: {best_valid_loss}\nCurr: {valid_loss}')
        if stale_epochs >= args.patience:
            print('Early stopping after %i stale epochs'%args.patience)
            break

    # save state when training ends for checking
    train_status = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'epoch': epoch+1,
        'lr': optimizer.param_groups[0]['lr'],
        'gae_train_true_emd': gae_train_true_emd,
        'gae_valid_true_emd': gae_valid_true_emd
    }
    torch.save(train_status, osp.join(save_dir, 'debug.pt'))

    # model training done
    train_epochs = list(range(epoch+1))
    early_stop_epoch = epoch - stale_epochs + 1
    loss_curves(train_epochs, early_stop_epoch, train_losses, valid_losses, save_dir, gae_train_true_emd, gae_valid_true_emd)
    # if args.train_emd_adversarially:
    #     emd_train_epochs = list(range(len(train_adv_loss)))
    #     adv_loss_curves(emd_train_epochs, train_adv_loss, valid_adv_loss, save_dir)

    # load best model
    del model
    torch.cuda.empty_cache()
    model = get_model(args.model, input_dim=input_dim, big_dim=big_dim, hidden_dim=hidden_dim, emd_modname=args.emd_model_name)
    model.load_state_dict(torch.load(modpath))
    if multi_gpu:
        model = DataParallel(model)
    model.to(device)

    inverse_standardization = args.standardize and args.plot_scale != 'standardized'
    plot_reco_for_loader(model, train_loader, device, scaler, inverse_standardization, osp.join(save_dir, 'reconstruction_post_train', 'train'), args.plot_scale)
    plot_reco_for_loader(model, valid_loader, device, scaler, inverse_standardization, osp.join(save_dir, 'reconstruction_post_train', 'valid'), args.plot_scale)
    plot_reco_for_loader(model, test_loader, device, scaler, inverse_standardization, osp.join(save_dir, 'reconstruction_post_train', 'test'), args.plot_scale)
    
    # plot emd correlation plots between emd-nn and true emd values between gae input and output
    if args.plot_emd_corr:
        if 'emd_loss' not in loss_ftn_obj.name:
            loss_ftn_obj = LossFunction('emd_loss', emd_model_name=args.emd_model_name, device=device)
        emd_loss_ftn = loss_ftn_obj.loss_ftn
        plot_emd_corr(model, train_loader, emd_loss_ftn, save_dir, 'emd_corr_train', scaler, device)
        plot_emd_corr(model, valid_loader, emd_loss_ftn, save_dir, 'emd_corr_valid', scaler, device)
        plot_emd_corr(model, test_loader, emd_loss_ftn, save_dir, 'emd_corr_test', scaler, device)
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
    parser.add_argument('--emd-nn-patience', type=int, help='patience for emd adv training', default=5, required=False)
    parser.add_argument('--epochs', type=int, help='max number of epochs', default=200, required=False)
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
    parser.add_argument("--plot-emd-corr-epoch", type=int, help="EMD Corr every n epochs (0 for no plots)", default=0, required=False)
    parser.add_argument("--train-emd-adversarially", action="store_true", help="train emd-nn loss function with (input, gae reco)", required=False)
    parser.add_argument("--drop-old-losses", action="store_true", help="don't load in old loss values", required=False)
    parser.add_argument("--plot-scale", choices=['cartesian','hadronic','standardized'],
                        help='classes for x-axis scaling for plotting reconstructions', default='cartesian', required=False)
    args = parser.parse_args()

    main(args)
