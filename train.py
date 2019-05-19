# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:14:47 2019

@author: souzac1
"""

import torch
import torch.nn as nn
import models
import utils
import math
#import matplotlib.pyplot as plt
#import cv2 as cv

# Specify the device used for training
device = torch.device('cuda:0')

# The dataset
crop_size          = [256, 256]
batch_size         = 5
train_dataset_path = "./data/paris_train_original/"
train_data         = utils.Data(train_dataset_path, random_crop = crop_size)
hole_size          = 100
n_iter             = math.ceil(train_data.n_images/batch_size)

print("Initializing training.")
# Generator and discriminator
G = models.Generator()
D = models.Discriminator(crop_size)

# Send the models to the specified device
D = D.to(device)
G = G.to(device)

# Path to save the models
save_path = "./saves/models_DG.pth"
restore = True
restore_path = "./saves/models_DG.pth"

# Create the model optimizers and loss function
d_optimizer  = torch.optim.Adam(D.parameters(), lr=0.0001)
g_optimizer  = torch.optim.Adam(G.parameters(), lr=0.00025)
criterion    = nn.BCELoss()
mse          = nn.MSELoss(reduction='mean')

train_with_D = True

# Number of epochs for training
n_epochs = 40

verbose_at_every = 3
save_at_every = 1

if restore:
    checkpoint = torch.load(restore_path)
    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])

print("Training initialized.")


running_mse_loss_last_history = [0]*n_iter
running_mse_loss_current_history = [0]*n_iter

for epoch in range(n_epochs):
    running_g_loss = 0
    all_images_processed = 0
    running_mse_loss = 0
    for iteration in range(n_iter):
        corrupted_data, actual_batch_size, locations, _, masks = train_data.get_batch(iteration + 1, batch_size, 
                                                                                      hole_size=hole_size, apply_hole = True, center = True)
        
        corrupted_data = utils.from_numpy_to_tensor(corrupted_data, device)
        masks = utils.from_numpy_to_tensor(masks, device)
                
        gt_data, _, _, _, _ = train_data.get_batch(iteration + 1, batch_size, apply_hole = False, center = False)
        gt_data = utils.from_numpy_to_tensor(gt_data, device)
        
        fake_gt_data, _ = G(corrupted_data)
        inverted_masks = (masks*-1) + 1

        if train_with_D:
            fake_gt_data_D = D(corrupted_data*inverted_masks + fake_gt_data*masks)
            real_gt_data_D = D(corrupted_data*inverted_masks + gt_data*masks)
            targets_real = torch.ones(actual_batch_size, 1).to(device)
            targets_fake = torch.zeros(actual_batch_size, 1).to(device)
            
            d_loss = criterion(fake_gt_data_D, targets_fake) + criterion(real_gt_data_D, targets_real)
            
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
        
        fake_gt_data, features_fake_gt_data  = G(corrupted_data)
        
        mse_loss     = mse(gt_data*masks, fake_gt_data*masks)
        g_loss       = mse_loss  
        
        if train_with_D:
            outputs      = D(corrupted_data*inverted_masks + fake_gt_data*masks)
            g_loss      += criterion(outputs, targets_real)
        
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        running_g_loss += g_loss.item()*actual_batch_size
        running_mse_loss += mse_loss.item()*actual_batch_size
        all_images_processed += actual_batch_size
        running_g_loss_ = running_g_loss/all_images_processed
        running_mse_loss_ = running_mse_loss/all_images_processed
        running_mse_loss_current_history[iteration] = running_mse_loss_
        
        if iteration % verbose_at_every == 0:
            if train_with_D:
                print("Epoch %d. Iteration %d. Generator loss: %f (MSE: %f). Discriminator loss: %f." % (epoch + 1, iteration + 1, g_loss.item(), mse_loss.item(), d_loss.item()))
            else:
                print("Epoch %d. Iteration %d. Generator loss: %f (MSE: %f)." % (epoch + 1, iteration + 1, g_loss.item(), mse_loss.item()))
            
            print("Running Generator Loss: %f. Running MSE Loss: %f. Last epoch's running MSE Loss: %f" % (running_g_loss_, running_mse_loss, running_mse_loss_last_history[iteration]))
    
    running_mse_loss_last_history = running_mse_loss_current_history.copy()
    running_mse_loss_current_history = [0]*n_iter
    if epoch % save_at_every == 0:
        torch.save({
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'Dopt_state_dict': d_optimizer.state_dict(),
        'Gopt_state_dict': g_optimizer.state_dict(),
        }, save_path)
        print("Models saved.")
print("Training completed")