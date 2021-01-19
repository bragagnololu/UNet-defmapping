# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:34:48 2020

@author: Lucimara Bragagnolo
"""

# scripts that must be in the same path that this one
from deforestation_mapping import *

# .GEOjson file of the area to be monitored
geojson_file = '/rondonia_square3.geojson'

# path to save the downloaded images
save_imgs = '/Downloaded'

# save RGB files
save_rgb = '/rgb_files'

# save tiles
save_tiles = '/tiles_imgs"

# Unet weights file
unet_weights = "/weights_file_of_trained_UNet.hdf5"

# Unet weights clouds file
unet_clouds = '/weights_file_of_clouds_trained_UNet.hdf5'

# classificated images path
class_path = "/predicted"

# classificated clouds images path
class_clouds = "/predicted_clouds"

# polygons save
poly_path = '/polygons'

# files saved after the trained UNet
percentiles_forest = ["/bands_third.npy",
                       "/bands_nin.npy"]

percentiles_clouds = ["/bands_third_clouds.npy",
                       "/bands_nin_clouds.npy"]

def_main(save_imgs, save_rgb, save_tiles, unet_weights, unet_clouds,
         class_path, class_clouds, poly_path, 
         percentiles_forest, percentiles_clouds, geojson_file)