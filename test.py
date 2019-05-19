# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:24:29 2019

@author: souzac1, azizazh
"""

import torch
import torch.nn as nn
import models
import utils
import math
import numpy as np

# Specify the device used for training
device = torch.device('cuda:0')

# The dataset
batch_size        = 8
test_dataset_path = "./data/paris_eval_gt/"
test_data         = utils.Data(test_dataset_path, resize=[227, 227], image_format = "png")
n_iter            = math.ceil(test_data.n_images/batch_size)

print("Initializing testing.")
# Generator and discriminator
G = models.Generator()

# Send the models to the specified device
G   = G.to(device)

mse = nn.MSELoss()

# Restoring the generator
restore_path = "./saves/models_DG.pth"
checkpoint = torch.load(restore_path)
G.load_state_dict(checkpoint['G_state_dict'])
G.eval()

# Path to save the results
save_path = "./output/"

verbose_at_every = 10

print("Testing initialized.")

running_g_loss = 0
mean_ssim = 0
mean_psnr = 0

chosen_image_names = []
for iteration in range(n_iter):
    # Gets corrupted data and mask
    corrupted_data, actual_batch_size, locations, crop_locations, masks = test_data.get_batch(iteration + 1, batch_size, hole_size=101,
                                                                                              apply_hole = True, center = True)
    corrupted_data = utils.from_numpy_to_tensor(corrupted_data, device)
    masks = utils.from_numpy_to_tensor(masks, device)
    
    # Gets ground truth data
    gt_data, _, _, _, _ = test_data.get_batch(iteration + 1, batch_size, apply_hole = False, center = False)
    gt_data = utils.from_numpy_to_tensor(gt_data, device)
    
    # Obtain reconstruction
    fake_gt_data, _ = G(corrupted_data)
    
    # Transform everything back to numpy array without normalization
    fake_remounted_gt_data = utils.remount_tensor_to_numpy(fake_gt_data, gt_data, locations)
    remounted_gt_data      = utils.remount_tensor_to_numpy(gt_data, gt_data, locations)
    remounted_corrupted_data = utils.remount_tensor_to_numpy(corrupted_data, corrupted_data, locations)
    
    # Calculates the metrics
    for i in range(0, len(fake_remounted_gt_data)):
        mean_ssim += utils.ssim(remounted_gt_data[i], fake_remounted_gt_data[i])
        mean_psnr += utils.psnr(remounted_gt_data[i], fake_remounted_gt_data[i])

    g_loss = mse(gt_data*masks, fake_gt_data*masks)
    running_g_loss += g_loss.item()*actual_batch_size
    
    # Saves the outputs
    image_names = test_data.get_batch_image_names(iteration + 1, batch_size)
    triple_images = np.concatenate((fake_remounted_gt_data, remounted_gt_data, corrupted_data), axis = 3)
    utils.save_numpy_batch(triple_images, save_path, image_names)
    utils.save_numpy_batch(fake_remounted_gt_data, save_path, image_names)
    
    
mean_g_loss = running_g_loss/test_data.n_images
mean_ssim   = mean_ssim/test_data.n_images
mean_psnr   = mean_psnr/test_data.n_images

print("Test ended. Mean G loss: %f. Mean SSIM: %f. Mean PSNR: %f" % (mean_g_loss, mean_ssim, mean_psnr))