import torch
import numpy as np
import energyflow as ef
import cv2

ONE_HUNDRED_GEV = 100.0
R = 0.4

# CV2 helpers
img_2d_coord_mat = torch.tensor(np.meshgrid(np.arange(28), np.arange(28))).T.reshape(-1,2)

def img_to_sig_cv2(arr):
    # create a signature to rep. img matrix for cv2 https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html#emd
    sig = torch.hstack((arr.flatten()[...,np.newaxis], img_2d_coord_mat))
    return sig.numpy()

def calc_emd_cv2(img1, img2):
    # calculate the emd between an image pair
    sig1 = img_to_sig_cv2(img1)
    sig2 = img_to_sig_cv2(img2)
    emd_val = cv2.EMD(sig1, sig2, cv2.DIST_L2)[0]
    return emd_val

def calc_emd_on_batch_cv2(img1, img2):
    # non-vectorized calculation between 2 batches
    emds = []
    for i1, i2 in zip(img1, img2):
        emd_val = calc_emd_cv2(i1, i2)
        emds.append(emd_val)
    return torch.tensor(emds)
  
  
# energyflow helpers
jet_coord_grid_step = R * 2 / 28
coord_axis = np.arange(-R, R, jet_coord_grid_step) + jet_coord_grid_step / 2
jet_img_coord = torch.tensor(np.meshgrid(coord_axis, coord_axis)).T.reshape(-1,2) # eta x phi for 28x28 grid

def img_to_part_array_ef(img):
    # make (pt, eta, phi) feat arr
    arr = torch.hstack((img.flatten().unsqueeze(1), jet_img_coord))
    return arr.numpy()

def calc_emd_ef(arr1, arr2):
    emd = ef.emd.emd(arr1, arr2, R=R)/ONE_HUNDRED_GEV
    return emd

def calc_emd_on_batch_ef(img1, img2):
    emds = []
    for i1, i2 in zip(img1, img2):
        a1, a2 = img_to_part_array_ef(i1), img_to_part_array_ef(i2)
        emd_val = calc_emd_ef(a1, a2)
        emds.append(emd_val)
    return torch.tensor(emds)
