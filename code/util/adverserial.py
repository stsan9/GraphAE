import torch
import energyflow as ef
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import to_dense_batch
from concurrent.futures import ThreadPoolExecutor

from itertools import chain
from util.loss_util import preprocess_emdnn_input

def train_emd_model(gae_model, emd_model, emd_optimizer, loader, scaler, num_procs=1, device=torch.device('cuda:0')):
    """
    Train emd model on (input, gae_reco) for one epoch.

    :param gae_model: pytorch in GAE
    :param emd_model: EMD-NN to train
    :param emd_optimizer: torch optimizer for emd_model
    :param loader: torch dataloader
    :param scaler: StandardScaler for reversing data to unstandardized form
    :param num_procs: how many threads to execute for emd calculation
    :param device: device model is on
    :return: average loss of EMD-NN
    """
    def calc_true_emd(jets1, jets2):
        """
        energyflow calculation of emd

        :param jets1: list of numpy arrays (n x [pt eta phi])
        :param jets2: list of numpy arrays (n x [pt eta phi])
        """
        true_emds = []
        for j1, j2 in zip(jets1, jets2):
            emd = ef.emd.emd(j1, j2)
            true_emds.append(emd)
        return true_emds

    sum_loss = 0

    t = tqdm(loader)
    for b in t:
        b = b.clone()
        b.to(device)

        with torch.no_grad():
            # get gae reconstruction
            jet_reco = gae_model(b)

        # inverse scaling
        jet_reco = scaler.inverse_transform(jet_reco)
        jet_in = scaler.inverse_transform(b.x)

        # numpy batches
        jet_reco[jet_reco < 0] = 0  # no negative pt to calc true emd
        jet_reco_np = to_dense_batch(x=jet_reco, batch=b.batch)[0].detach().cpu().numpy()
        jet_in_np = to_dense_batch(x=jet_in, batch=b.batch)[0].detach().cpu().numpy()

        # split data into chunks for multithreaded calculation of true emd
        jet_chunks_len = len(jet_in_np) // num_procs
        jet_chunks_in = []
        jet_chunks_reco = []
        for i in range(0, num_procs, jet_chunks_len):
            jet_chunks_in.append(jet_in_np[i:i + jet_chunks_len])
            jet_chunks_reco.append(jet_reco_np[i:i + jet_chunks_len])

        with ThreadPoolExecutor(max_workers=num_procs) as executor:
            emd_lists = executor.map(calc_true_emd, jet_chunks_in, jet_chunks_reco)
        emds = list(chain.from_iterable(emd_lists))

        # emd nn formatting
        if (not jet_reco.is_cuda) or (not jet_in.is_cuda):
            jet_reco.to(device)
            jet_in.to(device)
        emd_nn_inputs = preprocess_emdnn_input(jet_in, jet_reco, b.batch)
        emd_targets = torch.tensor(emds).to(device)

        # train emd network
        emd_preds = emd_model(emd_nn_inputs)[0] # index 0 to toss extra output
        emd_preds = emd_preds.squeeze()
        loss = F.mse_loss(emd_preds, emd_targets, reduction='mean')
        emd_optimizer.zero_grad()
        loss.backward()
        emd_optimizer.step()


        loss = loss.item()
        sum_loss += loss
        t.set_description('emd-nn train loss = %.7f' % loss)
        t.refresh() # to show immediately the update

    # return average loss of emd network during training
    return sum_loss / len(loader)
