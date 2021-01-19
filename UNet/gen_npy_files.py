# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:34:20 2020

@author: Xeon
"""
# generating .npy files of training and validation data

import numpy as np
import rasterio
import skimage.transform as trans

#MUDANDO VALIDATION
import os

nmber_channels = 4
# LOADING THE DATA
# TRAINING
train_path = '/training'
image_folder = 'image'
mask_folder = 'label'
target_size = (512,512)

image_list = []

names_images = sorted(os.listdir(os.path.join(train_path,image_folder)))
#print(names_slope)

for image in names_images:
    file_open = os.path.join(train_path,image_folder,image)
    #print(file_open)
    image_data = rasterio.open(file_open)
    image_array = image_data.read()
    
    if (np.size(image_array,1) - np.size(image_array,2)) != 0: # se nao for quadrado 512x512
        image_array = trans.resize(image_array,(nmber_channels,target_size[0],target_size[1]),preserve_range=True) # target size - into 0 and 1

    image_list.append(image_array)

image_array = np.stack(image_list)
image_array = np.transpose(image_array, (0,2,3,1)) # channels last


masks_list = []

names_masks = sorted(os.listdir(os.path.join(train_path,mask_folder)))
#print(names_slope)

for maskk in names_masks:
    file_open = os.path.join(train_path,mask_folder,maskk)
    #print(file_open)
    mask_data = rasterio.open(file_open)
    mask_array = mask_data.read()
    
    if (np.size(mask_array,1) - np.size(mask_array,2)) != 0:
        mask_array = trans.resize(mask_array,(1,target_size[0],target_size[1]),preserve_range=True) # target size
    mask_array[mask_array > 0.5] = 1 # ADICIONEI AQUI
    mask_array[mask_array <= 0.5] = 0 # ADICIONEI AQUI
    
    masks_list.append(mask_array)

mask_array = np.stack(masks_list)
mask_array = np.transpose(mask_array, (0,2,3,1)) # channels last

#--------------------------------------------------------------------------------
# VALIDATION
val_path = '/val'
val_image_folder = 'images'
val_mask_folder = 'masks'

image_list_val = []

val_names_images = sorted(os.listdir(os.path.join(val_path,val_image_folder)))

for image in val_names_images:
    file_open = os.path.join(val_path,val_image_folder,image)
    image_data = rasterio.open(file_open)
    val_image_array = image_data.read()
    
    if (np.size(val_image_array,1) - np.size(val_image_array,2)) != 0:
        val_image_array = trans.resize(val_image_array,(nmber_channels,target_size[0],target_size[1]),preserve_range=True) # target size

    image_list_val.append(val_image_array)

val_image_array = np.stack(image_list_val)
val_image_array = np.transpose(val_image_array, (0,2,3,1)) # channels last

val_slope_list = []


val_masks_list = []

val_names_masks = sorted(os.listdir(os.path.join(val_path,val_mask_folder)))
#print(names_slope)

for maskk in val_names_masks:
    file_open = os.path.join(val_path,val_mask_folder,maskk)
    #print(file_open)
    mask_data = rasterio.open(file_open)
    val_mask_array = mask_data.read()
    
    if (np.size(val_mask_array,1) - np.size(val_mask_array,2)) != 0:
        val_mask_array = trans.resize(val_mask_array,(1,target_size[0],target_size[1]),preserve_range=True) # target size
        
    val_mask_array[val_mask_array > 0.5] = 1 # ADICIONEI AQUI
    val_mask_array[val_mask_array <= 0.5] = 0 # ADICIONEI AQUI

    val_masks_list.append(val_mask_array)

val_mask_array = np.stack(val_masks_list)
val_mask_array = np.transpose(val_mask_array, (0,2,3,1)) # channels last
#-------------------------------------
