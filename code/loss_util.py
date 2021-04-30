import torch
"""TODO: turn into class instead"""

def chamfer_loss(x,y):
    nparts = x.shape[0]
    dist = torch.pow(torch.cdist(x,y),2)
    in_dist_out = torch.min(dist,dim=0)
    out_dist_in = torch.min(dist,dim=1)
    loss = torch.sum(in_dist_out.values + out_dist_in.values) / nparts
    return loss

# Reconstruction + KL divergence losses
def vae_loss(x, y, mu, logvar):
    BCE = chamfer_loss(x,y)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def emd(x, y, emd_model):
    model.eval()
    # concatenate column of 1s to one jet and -1 to other jet
    x = torch.cat((x,torch.ones(len(x),1)), 1)
    y = torch.cat((y,torch.ones(len(y),1) * -1), 1)
    jet_pair = torch.cat((x,y),0)
    # get emd between x and y
    emd = emd_model(jet_pair)
    return emd