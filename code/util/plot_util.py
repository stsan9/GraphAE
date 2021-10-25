import torch
import numpy as np
import mplhep as hep
import os.path as osp
import energyflow as ef
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from torch_geometric.utils import to_dense_batch

plt.style.use(hep.style.CMS)

def loss_distr(losses, save_name):
    """
        Plot distribution of losses
    """
    plt.figure(figsize=(6,4.4))
    plt.hist(losses,bins=np.linspace(0, 600, 101))
    plt.xlabel('Loss', fontsize=16)
    plt.ylabel('Jets', fontsize=16)
    plt.savefig(osp.join(save_name+'.pdf'))
    plt.close()

def plot_reco_difference(input_fts, reco_fts, save_path, feature='hadronic'):
    """
    Plot the difference between the autoencoder's reconstruction and the original input

    Args:
        input_fts (numpy array): the original features of the particles
        reco_fts (numpy array): the reconstructed features
        model_fname (str): name of saved model
    """
    if isinstance(input_fts, torch.Tensor):
        input_fts = input_fts.numpy()
    if isinstance(reco_fts, torch.Tensor):
        reco_fts = reco_fts.numpy()

    Path(save_path).mkdir(parents=True, exist_ok=True)
    label = ['$p_x~[GeV]$', '$p_y~[GeV]$', '$p_z~[GeV]$']
    feat = ['px', 'py', 'pz']
    if feature == 'hadronic' or feature == 'standardized':
        label = ['$p_T$', '$eta$', '$phi$']
        feat = ['pt', 'eta', 'phi']

    # make a separate plot for each feature
    for i in range(input_fts.shape[1]):
        plt.style.use(hep.style.CMS)
        plt.figure(figsize=(10,8))
        if feature == 'cartesian':
            bins = np.linspace(-20, 20, 101)
            if i == 3:  # different bin size for E momentum
                bins = np.linspace(-5, 35, 101)
        elif feature == 'hadronic':
            bins = np.linspace(-2, 2, 101)
            if i == 0:  # different bin size for pt rel
                bins = np.linspace(-0.05, 0.1, 101)
        else:
            bins = np.linspace(-1, 1, 101)
        plt.ticklabel_format(useMathText=True)
        plt.hist(input_fts[:,i], bins=bins, alpha=0.5, label='Input', histtype='step', lw=5)
        plt.hist(reco_fts[:,i], bins=bins, alpha=0.5, label='Output', histtype='step', lw=5)
        plt.legend(title='QCD dataset', fontsize='x-large')
        plt.xlabel(label[i], fontsize='x-large')
        plt.ylabel('Particles', fontsize='x-large')
        plt.tight_layout()
        plt.savefig(osp.join(save_path, feat[i] + '.pdf'))
        plt.close()

@torch.no_grad()
def gen_in_out(model, loader, device):
    model.eval()
    input_fts = []
    reco_fts = []

    for t in loader:
        if isinstance(t, list):
            for d in t:
                input_fts.append(d.x)
        else:
            input_fts.append(t.x)
            t.to(device)

        reco_out = model(t)
        if isinstance(reco_out, tuple):
            reco_out = reco_out[0]
        reco_fts.append(reco_out.cpu().detach())

    input_fts = torch.cat(input_fts)
    reco_fts = torch.cat(reco_fts)
    return input_fts, reco_fts

def plot_reco_for_loader(model, loader, device, scaler, inverse_scale, save_dir, feature_format):
    input_fts, reco_fts = gen_in_out(model, loader, device)
    if inverse_scale:
        input_fts = scaler.inverse_transform(input_fts)
        reco_fts = scaler.inverse_transform(reco_fts)
    plot_reco_difference(input_fts, reco_fts, save_dir, feature_format)


def loss_curves(epochs, early_stop_epoch, train_loss, valid_loss, save_path, train_true_emd=None, valid_true_emd=None):
    '''
        Graph our training and validation losses.
    '''
    if train_true_emd == None or valid_true_emd == None:
        plt.plot(epochs, train_loss, valid_loss)
        plt.xticks(epochs)
        ax = plt.gca()
        ax.set_yscale('log')
        if max(epochs) < 60:
            ax.locator_params(nbins=10, axis='x')
        else:
            ax.set_xticks(np.arange(0, max(epochs), 20))
        if early_stop_epoch != None:
            plt.axvline(x=early_stop_epoch, linestyle='--')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(['Train', 'Validation', 'Best model'])
    else:
        plt.plot(epochs, train_loss, label='Pred EMD (Train)')
        plt.plot(epochs, valid_loss, label='Pred EMD (Valid)')
        plt.plot(epochs, train_true_emd, label='True EMD (Train)')
        plt.plot(epochs, valid_true_emd, label='True EMD (Valid.)')
        plt.xticks(epochs)
        ax = plt.gca()
        ax.set_yscale('log')
        if max(epochs) < 60:
            ax.locator_params(nbins=10, axis='x')
        else:
            ax.set_xticks(np.arange(0, max(epochs), 20))
        if early_stop_epoch != None:
            plt.axvline(x=early_stop_epoch, linestyle='--', label='Best model')
        plt.xlabel("Epochs")
        plt.ylabel("EMD-NN Loss")
        plt.legend()
    plt.savefig(osp.join(save_path, 'loss_curves.pdf'))
    plt.savefig(osp.join(save_path, 'loss_curves.png'))
    plt.close()

def adv_loss_curves(epochs, train_loss, valid_loss, save_path):
    '''
        Graph our training and validation losses.
    '''
    plt.plot(epochs, train_loss, valid_loss)
    plt.xticks(epochs)
    ax = plt.gca()
    ax.set_yscale('log')
    if max(epochs) < 60:
        ax.locator_params(nbins=10, axis='x')
    else:
        ax.set_xticks(np.arange(0, max(epochs), 20))
    plt.title("EMD Model Adversarial Training Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend(['Train', 'Validation'])
    plt.savefig(osp.join(save_path, 'emd_adv_loss_curves.pdf'))
    plt.savefig(osp.join(save_path, 'emd_adv_loss_curves.png'))
    plt.close()

def epoch_emd_corr(in_parts, gen_parts, pred_emd, save_dir, epoch):
    """
    Plot emd correlation plot at this epoch

    :param in_parts: (batch, n, feat) torch tensor input jets
    :param gen_parts: (batch, n, feat) torch tensor for model reco
    :param pred_emd: list of predicted emds from emd-nn
    :param save_dir: directory to save plot
    :param epoch: epoch correlation plot is made for
    """

    save_dir = osp.join(save_dir, 'emd_corr_plots')
    Path(save_dir).mkdir(exist_ok=True)

    true_emd = []
    for x, y in zip(in_parts, gen_parts):
        emd = ef.emd.emd(x, y, n_iter_max=500000, return_flow=False, norm=True)
        true_emd.append(emd)
    true_emd = np.array(true_emd)
    pred_emd = np.array(pred_emd)
    np.save(osp.join(save_dir, f'true_emd_{epoch}'), true_emd)
    np.save(osp.join(save_dir, f'pred_emd_{epoch}'), pred_emd)

    # plot figures
    plt.rcParams['figure.figsize'] = (4,4)
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['font.family'] = 'serif'

    max_range = 0.8

    fig, ax = plt.subplots(figsize =(5, 5)) 
    plt.hist(true_emd, bins=np.linspace(0, max_range, 101), label='True', alpha=0.5)
    plt.hist(pred_emd, bins=np.linspace(0, max_range, 101), label = 'Pred.', alpha=0.5)
    plt.legend()
    ax.set_xlabel('EMD [GeV]') 
    fig.savefig(osp.join(save_dir,f'EMD_ep_{epoch}.pdf'))
    fig.savefig(osp.join(save_dir,f'EMD_ep_{epoch}.png'))

    fig, ax = plt.subplots(figsize =(5, 5)) 
    x_bins = np.linspace(0, max_range, 101)
    y_bins = np.linspace(0, max_range, 101)
    plt.hist2d(true_emd, pred_emd, bins=[x_bins,y_bins])
    ax.set_xlabel('True EMD [GeV]')  
    ax.set_ylabel('Pred. EMD [GeV]')
    fig.savefig(osp.join(save_dir,f'EMD_corr_ep_{epoch}.pdf'))
    fig.savefig(osp.join(save_dir,f'EMD_corr_ep_{epoch}.png'))

def plot_jet_images(jet, save_dir, save_name):
    """
    :param jet: np array [pt, y, phi]
    """
    if isinstance(jet, torch.Tensor):
        jet = jet.numpy()

    plt.imshow(ef.utils.pixelate(jet))
    plt.savefig(osp.join(save_dir, save_name))
    plt.close()

def reco_relative_diff(jet_in, jet_out, save_dir, save_name):
    """
    Plot relative difference between input and predicted features.
    Assumes standard distribution

    :param jet_in: np array [pt, y, phi]
    :param jet_out: np array [pt, y, phi]
    """
    if isinstance(jet_in, torch.Tensor):
        jet_in = jet_in.numpy()
    if isinstance(jet_out, torch.Tensor):
        jet_out = jet_out.numpy()

    rel_diff = (jet_out - jet_in) / (jet_in + 1e-12)

    bins = np.linspace(-1,1, 30)

    plt.hist(rel_diff[:,0], bins=bins)
    feat = 'p_T'
    plt.title(feat)
    plt.savefig(osp.join(save_dir, save_name + '_' + feat))
    plt.close()

    plt.hist(rel_diff[:,1], bins=bins)
    feat = 'eta'
    plt.title(feat)
    plt.savefig(osp.join(save_dir, save_name + '_' + feat))
    plt.close()

    plt.hist(rel_diff[:,2], bins=bins)
    feat = 'phi'
    plt.title(feat)
    plt.savefig(osp.join(save_dir, save_name + '_' + feat))
    plt.close()

@torch.no_grad()
def plot_emd_corr(model, loader, emd_loss_ftn, save_dir, save_name, scaler, device):
    """
    :param model: GAE
    :param loader: torch dataloader
    :param emd_loss_ftn: function for predicting emd using nn
    :param save_dir: where to save plots
    :param save_name: name to save plot as
    :param scaler: prefitted Standardizer
    :param device: torch device cpu/gpu
    """
    def make_plots(true_emd, pred_emd):
        """
        :param true_emd: np array
        :param pred_emd: np array
        """
        # plot figures
        plt.rcParams['figure.figsize'] = (4,4)
        plt.rcParams['figure.dpi'] = 120
        plt.rcParams['font.family'] = 'serif'

        max_range = max(np.max(true_emd), np.max(pred_emd))
        fig, ax = plt.subplots(figsize =(9, 9)) 
        plt.hist(true_emd, bins=np.linspace(0, max_range , 31),label='True', alpha=0.5)
        plt.hist(pred_emd, bins=np.linspace(0, max_range, 31),label = 'Pred.', alpha=0.5)
        plt.legend()
        ax.set_xlabel('EMD') 
        fig.savefig(osp.join(save_dir, save_name + '2'))

        fig, ax = plt.subplots(figsize =(9, 9))
        x_bins = np.linspace(0, max_range, 31)
        y_bins = np.linspace(0, max_range, 31)
        plt.hist2d(true_emd, pred_emd, bins=[x_bins, y_bins])
        ax.set_xlabel('True EMD')  
        ax.set_ylabel('Pred. EMD')
        plt.savefig(osp.join(save_dir, save_name))
        plt.close()

    pred_emd = []
    true_emd = []
    model.eval()
    for b in tqdm(loader, desc='Calculating True/Pred. EMDs'):
        b = b.clone()
        b.to(device)

        # get gae reconstruction
        jet_reco = model(b)

        # inverse scales to feed into emd network
        jet_reco = scaler.inverse_transform(jet_reco)
        jet_in = scaler.inverse_transform(b.x)
        pred = emd_loss_ftn(jet_in, jet_reco, b.batch, mean=False).reshape(-1).tolist()
        pred_emd += pred

        # calc true emd
        jet_reco[jet_reco < 0] = 0  # no negative pt to calc true emd
        jet_reco = to_dense_batch(x=jet_reco, batch=b.batch)[0].detach().cpu().numpy()
        jet_in = to_dense_batch(x=jet_in, batch=b.batch)[0].detach().cpu().numpy()
        for j1, j2 in zip(jet_reco, jet_in):
            emd_val = ef.emd.emd(j1, j2)
            true_emd.append(emd_val)
    make_plots(true_emd, pred_emd)
