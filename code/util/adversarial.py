import torch
import energyflow as ef
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import to_dense_batch
from concurrent.futures import ThreadPoolExecutor

from itertools import chain
from util.loss_util import preprocess_emdnn_input
from util.plot_util import plot_emd_training_one_batch

def loop_emd_model(gae_model, emd_model, emd_optimizer, batch, scaler, max_iters=1, patience=3, batch_num=0, save_path='.', device=torch.device('cuda:0')):
    """
    Train emd model on (input, gae_reco) for one epoch. Can use for validation if emd_optimizer is None.

    :param gae_model: GAE pytorch model to generate reco input
    :param emd_model: EMD-NN to train
    :param emd_optimizer: torch optimizer for emd_model; None for validation
    :param batch: the batch we want to optimize on
    :param scaler: StandardScaler for reversing data to unstandardized form
    :param max_iters: how many iterations to loop
    :param patience: train emd patience
    :param batch_num: what batch is being trained on
    :param save_path: path to save any figures
    :param device: device model is on
    :return: average loss of EMD-NN
    """
    def calc_true_emd(jet1, jet2):
        """
        energyflow calculation of emd

        :param jet1: numpy array (n x [pt eta phi])
        :param jet2: numpy array (n x [pt eta phi])
        """
        emd = ef.emd.emd(jet1, jet2, n_iter_max=5000000)
        return emd

    sum_loss = 0

    b = batch
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

    # emd inference
    if emd_optimizer == None:   # validate
        with torch.no_grad():
            emd_preds = emd_model(emd_nn_inputs)[0]
    else:   # train
        losses = []
        emd_diffs = []
        stale_epochs = 0
        best_loss = 9999999
        t = tqdm(range(max_iters), leave=False)
        for _ in t:
            emd_preds = emd_model(emd_nn_inputs)[0] # index 0 to toss extra output
            emd_preds = emd_preds.squeeze()

            loss = F.mse_loss(emd_preds, emd_targets, reduction='mean')
            emd_diffs.append(torch.mean(torch.abs(emd_targets - emd_preds)).item())
            if loss <= best_loss:   # update weights
                stale_epochs = 0

                if emd_optimizer != None:
                    emd_optimizer.zero_grad()
                    loss.backward()
                    emd_optimizer.step()

                losses.append(loss.item())
                if ((emd_preds - emd_targets) / emd_targets).mean() <= 0.05:  # another early stop condition
                    break
            else:   # exceeds patience
                losses.append(loss.item())
                stale_epochs += 1
                if stale_epochs == patience:
                    break
            t.set_description('EMD-NN train loss = %.7f' % loss)
            t.refresh()

        sum_loss = sum(losses)

        if batch_num < 10:
            plot_emd_training_one_batch(range(len(losses)), losses, emd_diffs, save_name=f'batch_{batch_num}', save_path=save_path)

    if emd_optimizer == None:
        print('EMD-NN valid loss = %.7f' % loss)

    return sum_loss, emd_preds, emd_targets
