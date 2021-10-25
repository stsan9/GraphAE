import torch
import energyflow as ef
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import to_dense_batch
from concurrent.futures import ThreadPoolExecutor

from itertools import chain
from util.loss_util import preprocess_emdnn_input

def loop_emd_model(gae_model, emd_model, emd_optimizer, loader, scaler, device=torch.device('cuda:0')):
    """
    Train emd model on (input, gae_reco) for one epoch. Can use for validation if emd_optimizer is None.

    :param gae_model: GAE pytorch model to generate reco input
    :param emd_model: EMD-NN to train
    :param emd_optimizer: torch optimizer for emd_model; None for validation
    :param loader: torch dataloader
    :param scaler: StandardScaler for reversing data to unstandardized form
    :param device: device model is on
    :return: average loss of EMD-NN
    """
    def calc_true_emd(jet1, jet2):
        """
        energyflow calculation of emd

        :param jet1: numpy array (n x [pt eta phi])
        :param jet2: numpy array (n x [pt eta phi])
        """
        emd = ef.emd.emd(jet1, jet2)
        return emd

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

        with ThreadPoolExecutor() as executor:
            emds = executor.map(calc_true_emd, jet_in_np, jet_reco_np)
        emds = [e for e in emds]

        # emd nn input formatting
        if (not jet_reco.is_cuda) or (not jet_in.is_cuda):
            jet_reco.to(device)
            jet_in.to(device)
        emd_nn_inputs = preprocess_emdnn_input(jet_in, jet_reco, b.batch)
        emd_targets = torch.tensor(emds).to(device)

        # emd network
        if emd_optimizer == None:
            with torch.no_grad():
                emd_preds = emd_model(emd_nn_inputs)[0] # index 0 to toss extra output
        else:
            emd_preds = emd_model(emd_nn_inputs)[0] # index 0 to toss extra output
        emd_preds = emd_preds.squeeze()

        loss = F.mse_loss(emd_preds, emd_targets, reduction='mean')
        if emd_optimizer != None:
            emd_optimizer.zero_grad()
            loss.backward()
            emd_optimizer.step()

        loss = loss.item()
        sum_loss += loss
        if emd_optimizer == None:
            t.set_description('EMD-NN valid loss = %.7f' % loss)
        else:
            t.set_description('EMD-NN train loss = %.7f' % loss)
        t.refresh() # to show immediately the update

    # return average loss of emd network during training
    return sum_loss / len(loader)
