import numpy as np
import energyflow as ef
import torch
import jetnet
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from qg_data_to_img import get_qg_imgs

device = torch.device('cuda:0')

# DEFINE MODELS
class EMDCNN(nn.Module):
    def __init__(self):
        super(EMDCNN, self).__init__()
        
        self.convs = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding='same')
        )
        self.fcn = nn.Sequential(
            nn.Linear(28 * 28 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, img1, img2):
        img = torch.cat([img1, img2], dim=1)

        out = self.convs(img)
        out = out.flatten(start_dim=1) # flatten everything excluding batch dimension
        out = self.fcn(out)
        return out
    
class SymmEMDCNN(nn.Module):
    def __init__(self):
        super(SymmEMDCNN, self).__init__()
        self.emdcnn = EMDCNN()
        
    def forward(self, img1, img2):
        out1 = self.emdcnn(img1, img2)
        out2 = self.emdcnn(img2, img1)
        avg = (out1 + out2) / 2
        return nn.functional.softplus(avg)

# get data
train_imgs, valid_imgs = get_qg_imgs()

# initialize model
lr = 0.001
emdcnn = SymmEMDCNN().to(device)
optimizer = torch.optim.Adam(emdcnn.parameters(), lr=lr)

# PRETRAIN EMD-CNN
max_epochs = 50
batch_size = 256

patience = 10
stale_epochs = 0

train_loss = []
valid_loss = []

for epoch in tqdm(range(max_epochs)):

    # TRAIN - take a batch, and pair with every other batch in the dataset
    avg_loss = 0
    rand_indices = torch.randperm(len(train_imgs)) # shuffling mechanism
    t = tqdm(range(0, len(train_imgs) - batch_size, batch_size), leave=False, desc='Training...')
    emdcnn.train()
    for i in t:
        batch_indices = rand_indices[i : i + batch_size]
        batch = train_imgs[batch_indices]
        batch_cpu_copy = batch.clone()

        inner_t = tqdm(range(i + batch_size, len(train_imgs), batch_size), leave=False, desc='Inner-Loop...')
        for j in inner_t:
            batch_indices_2 = rand_indices[j : j + batch_size]
            batch_2 = train_imgs[batch_indices_2]
            if len(batch_2) != len(batch):
                continue

            true_emds = calc_emd_on_batch_cv2(batch_cpu_copy, batch_2).to(device)

            batch = batch.to(device)
            batch_2 = batch_2.to(device)

            model_output = emdcnn(batch, batch_2)

            loss = nn.functional.mse_loss(true_emds, model_output.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.sum().item()
            loss += loss / len(batch)
        loss /= len(inner_t)
        t.set_description('train loss = %.7f' % loss)
        t.refresh()
        avg_loss += loss / (len(t) - 1)

    train_loss.append(avg_loss)

    if avg_loss > max(train_loss):
        stale_epochs += 10
        if state_epochs == patience:
            break
    else:
        stale_epochs = 0
        torch.save(emdcnn, '/anomalyvol/qg_adv_experiments/emdcnn.pt')
