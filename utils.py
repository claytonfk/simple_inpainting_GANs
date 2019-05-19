# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:30:11 2019

@author: souzac1
"""

import glob
import cv2 as cv
import random
import math
import numpy as np
from skimage.measure import compare_ssim
import warnings
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)

# Data class
class Data:
    def __init__(self, data_path, image_format='JPG', shuffle = True, normalize = True, 
                 resize = -1, random_crop = -1):
        ''' Constructor of the data class with methods and attributes
        related to loading images and applying operations to them
        
        Args:
            data_path: the path to the folder where the images are contained
            image_format: format of the images in data_path. Default: png
            shuffle: whether or not to shuffle the data. Default: True
            normalize: whether or not to normalize the data. Default: True
            resize: whether or not to resize the images. Default: -1, [resized_height, resized_width] if not
            random_crop: whether or not to random crop the images. Default: -1, [crop_height, crop_width] if not
        ''' 
        
        print("Initializing dataset.")
        self.data_path = data_path
        self.image_format = image_format
        self.normalize = normalize
        self.images_paths = glob.glob(self.data_path + '*')
        self.images_paths = [e for e in self.images_paths if self.image_format in e]
        
        assert len(self.images_paths) > 0, "No images found."
        self.n_images = len(self.images_paths)
        self.resize = resize
        self.random_crop = random_crop

        if shuffle:
            random.shuffle(self.images_paths)
            
        print("Dataset initialized.")
        
    def random_shuffle(self):
        ''' Randomly shuffles the data
        
        '''
        random.shuffle(self.images_paths)
        
    def load_image(self, image_path, skip_normalization=False, skip_cropping=False):
        ''' Loads an image as numpy array 
        
        Args:
            image_path: the path to the image to be loaded
            skip_normalization: whether or not to ignore self.normalize flag. Default: False
            skip_cropping: whether or not to ignore self.random_crop flag. Default: False
            
        Returns:
            image: numpy array of shape [height, width, 3] in RGB. Type: uint8
        '''
        
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        assert image is not None, "Image could not be loaded."
        image = image[..., ::-1]
        crop_locations = -1
        
        if self.resize != -1:
            resized_height, resized_width = self.resize
            image = cv.resize(image,(resized_width,resized_height))
            
        if self.normalize and not skip_normalization:
            image = self.apply_normalization(image)
        
        if self.random_crop != -1 and not skip_cropping:
            cropped_height, cropped_width = self.random_crop
            h, w, _ = image.shape
            hi = math.ceil((h - cropped_height)/2)
            he = hi + cropped_height
            
            wi = math.ceil((w - cropped_width)/2)
            we = wi + cropped_width
            
            image = image[hi:he, wi:we, :]
            crop_locations = [hi, he, wi, we]
            
        return image, crop_locations
    
    def create_hole(self, image, hole_size=101, center=True):
        ''' Creates a white square hole with sides (in pixels) equal to hole_size
        
        Args:
            image: the image where the hole will be put 
                   (numpy array of shape [height, width, 3])
            hole_size: length of the square hole size (in pixels). Integer
                       Default: 101 pixels
            center: True -> to place the hole in the center of the image
                    False -> to place the hole at a random location
                    Default: True
            
        Returns:
            image: the image with the hole. (numpy array of shape [height, width, 3])
            location: the location of the hole as a python list [initial_height, final_height, initial_width, final_width]
        '''
        h,w,_ = image.shape
        assert hole_size < h, "Hole size is bigger than or equal to the height of the image"
        assert hole_size < w, "Hole size is bigger than or equal to the width of the image"
        
        if center:
            hi = math.ceil((h - hole_size)/2)
            he = hi + hole_size
            
            wi = math.ceil((w - hole_size)/2)
            we = wi + hole_size
        else:
            h_lim = h - hole_size - 1
            w_lim = w - hole_size - 1
            
            hi = random.randint(0, h_lim)
            he = hi + hole_size
            wi = random.randint(0, w_lim)
            we = wi + hole_size
            
        image[hi:he, wi:we, :] = [1, 1, 1]
        
        location = [hi, he, wi, we]
        
        return image, location
    
    def calculate_mean_std(self):
        ''' Calculates the means and standard deviations for all images in self.images_path

        Returns:
            mean: means (numpy array of shape [height, width, 3])
            std: standard deviations (numpy array of shape [height, width, 3])
        '''
        for i, image_path in enumerate(self.images_paths):
            image, _ = self.load_image(image_path, skip_normalization=True, skip_cropping=True)
            if i == 0:
                total_sum = np.zeros(image.shape)
            
            total_sum = total_sum + image
                
        self.mean = total_sum/self.n_images
        
        for i, image_path in enumerate(self.images_paths):
            image, _ = self.load_image(image_path, skip_normalization=True, skip_cropping=True)
            if i == 0:
                total_sum = np.zeros(image.shape)

            total_sum = total_sum + (image-self.mean)**2
            
        std = total_sum/self.n_images
        self.std = np.sqrt(std)

        return self.mean, self.std
    
    def get_batch(self, iteration, batch_size, apply_hole=False, hole_size=101, center=True):
        ''' Returns a batch of images contained in the data class 
        
            Args:
                iteration: index of the iteration (starts with 1)
                batch_size: size of the batch
                apply_hole: whether or not to make a hole in the images
                hole_size: hole size (only used if apply_hole=True)
                center: True -> to place the hole in the center of the image
                        False -> to place the hole at a random location
                        Default: True
                        Only used if apply_hole=True
        
            Returns:
                batch: numpy array contained the images [batch_size, height, width, 3]
                actual_batch_size: actual batch size
                locations: locations where the holes were created
                crop_locations: locations where the image was cropped
                masks: masks of the holes
        '''
        
        indices = [e + batch_size*(iteration-1) for e in range(0, batch_size)]
        
        if batch_size == 1:
            if indices[0] < self.n_images:
                image, crop_location = self.load_image(self.images_paths[indices[0]])
                mask = np.zeros(image.shape)
                if apply_hole:
                    image, location = self.create_hole(image,hole_size,center)
                    hi, he, wi, we = location
                    mask[hi:he, wi:we, :] = 1
                    locations = np.expand_dims(np.array(location), axis = 0)
                else:
                    locations = -1
                crop_locations = np.expand_dims(np.array(crop_location), axis = 0)
                masks = np.expand_dims(np.array(mask), axis = 0)
                batch = np.expand_dims(image, axis = 0)
        else:
            batch = []
            locations = []
            crop_locations = []
            masks = []
            for i in indices:
                if i < self.n_images:
                    image, crop_location = self.load_image(self.images_paths[i])
                    mask = np.zeros(image.shape)
                    if apply_hole:
                        image, location = self.create_hole(image,hole_size,center)
                        locations.append(location)
                        hi, he, wi, we = location
                        mask[hi:he, wi:we, :] = 1
                    else:
                        locations = -1
                    
                    masks.append(mask)
                    crop_locations.append(crop_location)
                    image = np.expand_dims(image, axis = 0)
                    batch.append(image)
                    
            locations = np.array(locations)
            batch = np.concatenate(batch, axis=0)*1.
            masks = np.expand_dims(masks, axis = 0)
            masks = np.concatenate(masks, axis=0)
        
        actual_batch_size = len(batch)
        
        return batch, actual_batch_size, locations, crop_locations, masks
    
    def get_batch_image_names(self, iteration, batch_size):
        ''' Returns the name of the images in the batch
        
            Args:
                iteration: index of the iteration (starts with 1)
                batch_size: size of the batch
            
            Returns:
                file_names: a list with the names of the files containing in the batch
        
        '''
        indices = [e + batch_size*(iteration-1) for e in range(0, batch_size)]
        file_names = []
        data_path_len = len(self.data_path)
        
        for i in indices:
            if i < self.n_images:
                file_names.append(self.images_paths[i][data_path_len:])
                
        return file_names
    
    def apply_normalization(self, image):
        ''' Performs normalization on an image dividing every pixel by 127.5 and subtracting 1
        
            Args:
                image: image in numpy array [height, width, 3])
        
            Returns:
                image: normalized image
        '''
        image = np.divide(image, 127.5) - 1
            
        return image
        
    
def psnr(gt_image, corrupted_image):
    ''' Calculates the peak signal to noise ratio 
    
    Args:
        gt_image: numpy array representing the ground truth (original) image 
        corrupted_image: numpy array representing the corrupted image
        
    Returns:
        psnr_value: the calculated peak signal to noise ratio
        
    '''
    mse = np.mean((corrupted_image.astype(float) - gt_image.astype(float))**2)
    
    if mse == 0:
        return 100
    
    return 20 * math.log10(255.0/math.sqrt(mse))

def ssim(gt_image, corrupted_image):
    ''' Calculates the structure similarity index
    
    Args:
        gt_image: numpy array representing the ground truth (original) image 
        corrupted_image: numpy array representing the corrupted image
        
    Returns:
        ssim_value: structure similarity index
        
    '''
    ssim_value = compare_ssim(gt_image, corrupted_image, data_range=corrupted_image.max() - corrupted_image.min(),
                              multichannel=True)
    return ssim_value  

def remove_normalization(batch):
    ''' Performs denormalization on a batch according to self.
    
        Args:
            batch: batch of images (numpy array of shape [batch_size, height, width, 3])
    
        Returns:
            batch: denormalized batch
    '''
    batch_size, _, _, _ = batch.shape
    
    for i in range(0, batch_size):
        batch[i] = np.multiply(batch[i] +1, 127.5) 
        
    return batch

def from_tensor_to_numpy(batch_tensor):
    ''' Transforms a Pytorch tensor to numpy array removing normalization 
    
    Args:
        batch_tensor: a pytorch tensor representing a batch of images [batch_size, 3, height, width]
        
    Returns:
        batch: denormalizated batch in numpy array uint8 [batch_size, height, width, 3]
    '''
    batch = batch_tensor.detach().cpu().transpose(1, 2).transpose(2, 3).numpy()
    batch = remove_normalization(batch)
    batch = batch.astype(np.uint8)
    
    return batch

def remount_tensor_to_numpy(fake_gt_tensor, gt_tensor, locations):
    fake_gt_numpy = from_tensor_to_numpy(fake_gt_tensor)
    gt_numpy = from_tensor_to_numpy(gt_tensor)
    remounted_gt_tensor = np.copy(gt_numpy)
    
    batch_size, _, _, _ = fake_gt_numpy.shape
    
    for i in range(0, batch_size):
        hi, he, wi, we = locations[i]
        remounted_gt_tensor[i, hi:he, wi:we, :] = fake_gt_numpy[i, hi:he, wi:we, :]
    
    
    return remounted_gt_tensor

def save_numpy_batch(numpy_batch, save_path, image_names):
    ''' Saves a numpy batch as images
    
    Args:
        numpy_batch: a numpy array [batch_size, height, width, 3]
        save_path: path to which the images will be saved
        image_names: a list containing the name of the images to be saved
    '''
    i = 0
    for image in numpy_batch:
        # RGB to BGR
        image = image[..., ::-1]
        cv.imwrite(save_path + image_names[i], image)
        i = i + 1
    
def from_numpy_to_tensor(np_array, device):
    ''' Transforms a numpy array to Pytorch tensor
    
    Args:
        np_array: a numpy array [batch_size, height, width, 3]
        device: the device to which the tensor is sent
        
    Returns:
        tensor: a pytorch tensor representing a batch of images [batch_size, 3, height, width]
    '''
    tensor = torch.from_numpy(np_array.copy()).to(device, dtype=torch.float)
    tensor = torch.transpose(tensor, 2, 3)
    tensor = torch.transpose(tensor, 1, 2)
    
    return tensor