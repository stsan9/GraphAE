import numpy as np
import energyflow as ef
import torch
import jetnet

def get_qg_imgs(num_data=20000):
    data = ef.qg_jets.load(num_data=num_data, pad=True, ncol=4, generator='pythia', with_bc=False, cache_dir='~/.energyflow')[0]

    data = data[...,:-1] # drop pid from column features

    # preprocess by centering jets and normalizing pts
    for x in data:
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()

    data_pre_img = data[:,:,[1,2,0]] # pt eta phi -> eta phi pt (order matters for to_image func)

    imgs = []
    for jet in data_pre_img:
        img = jetnet.utils.to_image(jet, im_size=28, maxR=0.4)
        if img.sum() > 0:
            imgs.append(img)

    imgs = np.stack(imgs)
    imgs = imgs[:, np.newaxis, ...] # add feature channel
    imgs = torch.tensor(imgs, dtype=torch.float32)

    # 80:20 train val split
    rand_inds = torch.randperm(len(imgs))
    thresh = int(0.8 * len(rand_inds))
    train_imgs = imgs[rand_inds[:thresh]]
    valid_imgs = imgs[rand_inds[thresh:]]
    
    return train_imgs, valid_imgs