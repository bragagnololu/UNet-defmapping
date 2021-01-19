# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:44:50 2020

@author: Lucimara Bragagnolo
"""

# scripts that must be in the same path that this one
import gdal_retile
from model_UNET_def import *
from data import *

import copy
import cv2
import datetime
import fiona
import glob
import numpy as np
import numpy_indexed as npi
import os
import os.path
from pylab import arange
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio import features
from scipy.ndimage import measurements
import skimage.io
from skimage import exposure
import shutil
import time
from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
import zipfile

import matplotlib.pyplot as plt

# download Sentinel-2 images
def download_images(save_imgs, save_rgb, save_tiles, unet_weights, unet_clouds,
                    class_path, class_clouds, poly_path, percentiles_forest, 
                    percentiles_clouds, boundsdata): 
    

    # connect to the API
    user = 'USERNAME'
    password = 'PASSWORD' 
    
    api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')
    
    # search by polygon
    footprint = geojson_to_wkt(read_geojson(boundsdata))
    
    # search for the images
    products = api.query(footprint,
                     date = (["NOW-30DAYS","NOW"]),
                     area_relation = 'IsWithin',
                     platformname = 'Sentinel-2',
                     processinglevel = 'Level-2A',
                     #cloudcoverpercentage = (0, 20)
                    )
    
    print(len(products))
    
    table_names = api.to_geodataframe(products)


    uuid_names = table_names['uuid'] # names to download
    name_zip = table_names['title'] #title of image
    extracted_name = table_names['filename'] # name of folder .SAFE
    
    k = 0
    
    # download images
    for fname in uuid_names:
        
        file_dir = save_imgs + '/' + extracted_name[k]

        
        if os.path.isdir(file_dir) is False:
            
            retval = os.getcwd()
            os.chdir(save_imgs)
            print("Downloading data...")
            api.get_product_odata(fname)
            api.download(fname)
            os.chdir(retval) # return to previous directory
            
            
            path_zip_name = save_imgs+'/'+name_zip[k]+'.zip'
            while not os.path.exists(path_zip_name):
                time.sleep(1)
                
            if os.path.isfile(path_zip_name):                  
                # extract files 
                zip_ref = zipfile.ZipFile(path_zip_name, 'r')
                zip_ref.extractall(save_imgs)
                zip_ref.close()
                os.remove(path_zip_name) # remove .zip file
                print("%s has been removed successfully" %name_zip[k]) 
                
                path_to_folder = save_imgs + '/' + extracted_name[k] + '/GRANULE/'
                
                # calls the rgb_tiles function
                dir_save_tiles = save_tiles+ '/' + name_zip[k]
                if os.path.isdir(dir_save_tiles) is False:
                        print('Creating RGB tiles')
                        os.mkdir(dir_save_tiles)
                        rgb_tiles(path_to_folder, save_rgb, dir_save_tiles, name_zip[k])
                
                # calls the application() Unet function
                save_class_path = class_path + '/' + name_zip[k]
                if os.path.isdir(save_class_path) is False:
                    print('Applying UNet')
                    os.mkdir(save_class_path)
                    application(dir_save_tiles, unet_weights, save_class_path, percentiles_forest, clouds = 0)

                    # merge predicted tiles into one GeoTiff                    
                    join_tiles(save_class_path,class_path, path_to_folder)
                    print("Tiles merged!")
                    
                save_class_clouds = class_clouds + '/' + name_zip[k]
                if os.path.isdir(save_class_clouds) is False:
                    print('Applying UNet clouds')
                    os.mkdir(save_class_clouds)
                    application(dir_save_tiles, unet_clouds, save_class_clouds, percentiles_clouds, clouds = 1)
                         
                    # merge the clouds predicted tiles into one GeoTiff
                    join_tiles(save_class_clouds, class_clouds, path_to_folder)
                    print("Clouds tiles merged!")
                
                # polygons evalutation
                print("Polygons evaluation")
                polygons(name_zip[k], class_path, class_clouds, path_to_folder, save_class_path, save_imgs, poly_path,time_spaced=None)
            
                k = k + 1
                
            else:
                raise ValueError("%s isn't a file!" %path_zip_name)
                
        else:
                path_to_folder = save_imgs + '/' + extracted_name[k] + '/GRANULE/'
                
                # calls the rgb_tiles function
                dir_save_tiles = save_tiles+ '/' + name_zip[k]
                if os.path.isdir(dir_save_tiles) is False:
                    print('Creating RGB tiles')
                    os.mkdir(dir_save_tiles)
                    rgb_tiles(path_to_folder, save_rgb, dir_save_tiles, name_zip[k])
                
                # calls the application() Unet function
                save_class_path = class_path + '/' + name_zip[k]
                if os.path.isdir(save_class_path) is False:
                    print('Applying UNet')
                    os.mkdir(save_class_path)
                    application(dir_save_tiles, unet_weights, save_class_path, percentiles_forest, clouds = 0)
            
                    # merge predicted tiles into one GeoTiff     
                    join_tiles(save_class_path,class_path, path_to_folder)
                    print("Tiles merged!")
                    
                save_class_clouds = class_clouds + '/' + name_zip[k]
                if os.path.isdir(save_class_clouds) is False:
                    print('Applying UNet clouds')
                    os.mkdir(save_class_clouds)
                    application(dir_save_tiles, unet_clouds, save_class_clouds,percentiles_clouds, clouds = 1)
                         
                    # merge the clouds predicted tiles into one GeoTiff
                    join_tiles(save_class_clouds, class_clouds, path_to_folder)
                    print("Clouds tiles merged!")               
                    
            
                # polygons evalutation
                print("Polygons evaluation")
                polygons(name_zip[k], class_path, class_clouds, path_to_folder, save_class_path, save_imgs, poly_path, time_spaced=None)
            
                k = k + 1
    
    return

def app_images_test(save_imgs, save_rgb, save_tiles, unet_weights, unet_clouds,
                    class_path, class_clouds, poly_path, time_spaced, 
                    percentiles_forest, percentiles_clouds, geojson_file=None):
        
    # dirpaths = []
    # dirnamess = []
    
    # for (dirpath, dirnames, filenames) in os.walk(save_imgs):
    #     dirpaths += [dirpath]
    #     dirnamess += dirnames
    
    # del dirpaths[0]
    
    # find directories
    searchcriteria = "*.SAFE"
    search = os.path.join(save_imgs, searchcriteria)
    
    dirpaths = glob.glob(search)
    # dirpaths.sort(key=os.path.getmtime, reverse=False) # sort files by date
    dirpaths.sort(key=os.path.getmtime, reverse=True)
    
    # testing the list elements positions
    img1 = dirpaths[0]
    img2 = dirpaths[-1]
    
    img1 = os.path.split(img1)
    img1 = img1[1]
    
    img2 = os.path.split(img2)
    img2 = img2[1]
    
    year1 = img1[11:15]
    month1 = img1[15:17]
    day1 = img1[17:19]
    
    year2 = img2[11:15]
    month2 = img2[15:17]
    day2 = img2[17:19]
    
    date1 = datetime.datetime(int(year1), int(month1), int(day1))
    date2 = datetime.datetime(int(year2), int(month2), int(day2)) 
    
    # if date1 < date2: # ALTERADO AQUI NO DIA 30-10 PARA TESTAR, ESTAVA INVERTIDO
    #     dirpaths.reverse() # invert elements from the list

    folders_name = list()
    
    for name_safe in dirpaths:
        if (name_safe.endswith('.SAFE')):
            folders_name.append(name_safe) # get the name of work directories
            
    for name_imgs in folders_name:
        
        path, extracted_name = os.path.split(name_imgs)
        name_zip = extracted_name.replace('.SAFE','')
    
        path_to_folder = save_imgs + '/' + extracted_name + '/GRANULE/'
                
        # calls the rgb_tiles function
        dir_save_tiles = save_tiles+ '/' + name_zip
        if os.path.isdir(dir_save_tiles) is False:
            print('Creating RGB tiles')
            os.mkdir(dir_save_tiles)
            rgb_tiles(path_to_folder, save_rgb, dir_save_tiles, name_zip)
        
        # calls the application() Unet function
        save_class_path = class_path + '/' + name_zip
        if os.path.isdir(save_class_path) is False:
            print('Applying UNet')
            os.mkdir(save_class_path)
            application(dir_save_tiles, unet_weights, save_class_path, percentiles_forest, clouds = 0)
        
            # merge predicted tiles into one GeoTiff     
            print("Tiles merged!")
            join_tiles(save_class_path,class_path, path_to_folder)
        
        save_class_clouds = class_clouds + '/' + name_zip
        if os.path.isdir(save_class_clouds) is False:
            print('Applying UNet clouds')
            os.mkdir(save_class_clouds)
            application(dir_save_tiles, unet_clouds, save_class_clouds, percentiles_clouds, clouds = 1)
                 
            # merge the clouds predicted tiles into one GeoTiff
            join_tiles(save_class_clouds, class_clouds, path_to_folder)
            print("Clouds tiles merged!")

        # polygons evalutation
        print("Polygons evaluation")
        polygons(name_zip, class_path, class_clouds, path_to_folder, save_class_path, save_imgs, poly_path, time_spaced)
        
    return 

def retile(inputList):
    tiler = gdal_retile
    tiler.Verbose = True
    #tiler.TileHeight = 512
    #tiler.TileWidth = 512
    tiler.TargetDir = inputList[1]
    #tiler.Levels = inputList[2]
    tiler.Names = inputList[0]
    tiler.CreateOptions = ["COMPRESS=LZW","TILED=YES"]
    tiler.main()
    
    return
    
def rgb_tiles(path_to_folder, save_rgb, save_tiles, name_rgb):
    
    #  open Bands 4, 3 and 2 with Rasterio
    listOfFiles = list()
    
    for (dirpath, dirnames, filenames) in os.walk(path_to_folder):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        
    for fname in listOfFiles:
        if (fname.endswith("B02_10m.jp2")):
            b2 = rasterio.open(fname)
            blue = b2.read(1)
        if (fname.endswith("B03_10m.jp2")):
            b3 = rasterio.open(fname)
            green = b3.read(1)
        if (fname.endswith("B04_10m.jp2")):
            b4 = rasterio.open(fname)
            red = b4.read(1)
        if (fname.endswith("B08_10m.jp2")):
            b8 = rasterio.open(fname)
            band8 = b8.read(1)

    # create an RGB image 
    path_to_rgb = save_rgb + '/' + name_rgb + '.tif'
    with rasterio.open(path_to_rgb,'w',driver='Gtiff', width=b4.width, height=b4.height, 
                  count=4,crs=b4.crs,transform=b4.transform, dtype='uint16') as rgb:
        rgb.write(red,1) 
        rgb.write(green,2) 
        rgb.write(blue,3) 
        rgb.write(band8,4)
        rgb.close()
    
    time.sleep(10)
    
    save_tiles = save_tiles + '//'

    inputList = [[path_to_rgb], save_tiles]

    retile(inputList)
    
    #shutil.move(path_to_rgb,save_rgb + '/already/' + name_rgb + '.tif')
    
    
    return

def application(save_tiles, unet_weights, class_path, percentiles, clouds):
    
    testGene = testGenerator(save_tiles,percentiles,clouds,target_size = (512,512),flag_multi_class = False,as_gray = False)
    
    if clouds == 1:
        model = unet(input_size = (512,512,3))
    else:
        model = unet(input_size = (512,512,4))
        
    model.load_weights(unet_weights) 
    
    listOfFiles = list()
    
    for (dirpath, dirnames, filenames) in os.walk(save_tiles):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        
    steps_application = len(listOfFiles)
    
    results = model.predict_generator(testGene,steps=steps_application, verbose=1)
    
    print(results.shape)
    
    #saveResult(tiles_dir,save_path,npyfile,flag_multi_class = False,num_class = 2)
    saveResult(save_tiles, class_path, results)
    
    return

def join_tiles(save_class_path,class_path, path_to_folder):
    
    # make a search criteria to select the files
    search_criteria = "S*.png"
    q = os.path.join(save_class_path, search_criteria)
    
    # list all files with glob() function
    dem_fps = glob.glob(q)
    print('Number of files to merge:',len(dem_fps))
    
    src_files_to_mosaic = []
    
    for fp in dem_fps:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
        
    # merge function returns a single mosaic array and the transformation info
    mosaic, out_trans = merge(src_files_to_mosaic)   
    
    # copy the metadata
    out_meta = src.meta.copy()
    
    # crs from a satellite image band
    listOfFiles = list()
    
    for (dirpath, dirnames, filenames) in os.walk(path_to_folder):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        
    for fname in listOfFiles:
        if (fname.endswith("B02_10m.jp2")):
            b2 = rasterio.open(fname)
            
    # metadata update
    out_meta.update({"driver": "GTiff",
                      "height": mosaic.shape[1],
                      "width": mosaic.shape[2],
                      "transform": out_trans,
                      "crs": b2.crs 
                      })
    
    # save the mosaic
    with rasterio.open(save_class_path + '_predicted.tif', "w", **out_meta) as dest:
        dest.write(mosaic)
        
    return

# load and interpolate cloud masks
def cloud_masks(path_to_folder, save_class_path):
    
    # crs from a satellite image band
    listOfFiles = list()
    
    for (dirpath, dirnames, filenames) in os.walk(path_to_folder):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        
    for fname in listOfFiles:
        if (fname.endswith("B02_10m.jp2")):
            #b2 = fname
            # read metadata information
            with rasterio.open(fname) as scl:
                B02 = scl.read()
                tmparr = np.empty_like(B02)
                aff = scl.transform
                print("Image shape:", B02.shape)
        
    # open the scl band
    for fname in listOfFiles:
        if (fname.endswith("SCL_20m.jp2")):
            #scl_file = fname        
            with rasterio.open(fname) as scl:
                BSCL=scl.read() 
                reproject(
                    BSCL, tmparr,
                    src_transform = scl.transform,
                    dst_transform = aff,
                    src_crs = scl.crs,
                    dst_crs = scl.crs,
                    resampling = Resampling.bilinear)
                BSCL = tmparr    
                print("SCL resized:", BSCL.shape)
        
    # create a new mask, just with the values for not be contabilized later (1, 2, 3, 6, 8, 9, 10, 11)
    BSCL = np.where(BSCL<=3, 9999, BSCL) 
    BSCL = np.where(BSCL==6, 9999, BSCL)
    BSCL = np.where((BSCL>=8) & (BSCL<12), 9999, BSCL)
    BSCL = np.where(BSCL!=9999, 0, BSCL)
    BSCL = BSCL.astype('uint16')
    

    #EXCLUIR DEPOIS
    # with rasterio.open(path_to_folder + '/_scl.tif','w',driver='Gtiff', width=scl.width, height=scl.height, 
    #                     count=1,crs=scl.crs,transform=scl.transform, dtype='uint16') as scl_10m:
    #     scl_10m.write(BSCL) 
    #     scl_10m.close()
        
    return BSCL

def evi(path_to_folder_evi):
    
    # Open Bands 8, 4 and 2 with rasterio
    listOfFiles = list()
    
    for (dirpath, dirnames, filenames) in os.walk(path_to_folder_evi):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        
    for fname in listOfFiles:
        if (fname.endswith("B08_10m.jp2")):
            b8 = rasterio.open(fname)
            B8 = b8.read()
        if (fname.endswith("B04_10m.jp2")):
            b4 = rasterio.open(fname)
            B4 = b4.read()
        if (fname.endswith("B02_10m.jp2")):
            b2 = rasterio.open(fname)
            B2 = b2.read()
            
    # Applying EVI equation
    EVI = 2.5*(B8 - B4)/((B8 + 6*B4 + 7.5*B2)+1)
    EVI[EVI > 1] = 1 
    EVI[EVI < -1] = -1
    
    # EXCLUIR DPS
    # with rasterio.open(path_to_folder_evi + '/_evi.tif','w',driver='Gtiff', width=b8.width, height=b8.height, 
    #                     count=1,crs=b8.crs,transform=b8.transform, dtype='float64') as scl_10m:
    #     scl_10m.write(EVI) 
    #     scl_10m.close()
    
    return EVI

def growth_rate(U0,U1,dt,Ku):
    # U0: matriz de EVI inicial
    # U1: matriz de EVI final
    # dt: intervalo de tempo entre U0 e U1
    # Ku: carrying capacity
    
    # Read size of stuff
    rows = np.size(U0,0)
    cols = np.size(U0,1)
    
    # Pre-allocation
    r_matrix = np.zeros([rows,cols]) # matrix with r values for each pixel
    r_mask = np.zeros([rows,cols]) # matrix indicanting negative r_values (negative growth)
    
    # Calculate r for each pixel
    difference = U0 - U1
    r_mask[difference == np.nan] = np.nan 
    r_matrix = -(1/dt)*np.log((Ku/U1-1)*U0/(Ku-U0))
    r_matrix[difference == np.nan] = np.nan
    r_matrix[r_matrix == np.inf] = np.nan
    r_matrix[r_matrix == - np.inf] = np.nan
    
    r_mask[r_matrix < 0] = 1 # Negative growth

    return r_matrix, r_mask

#polygons(name_zip[k], class_path)
def polygons(zip_name, class_path, class_clouds, path_to_folder, save_class_path, save_imgs, poly_path, time_spaced):
    
    print("Current: ",zip_name)
    # grid code
    grid_img = zip_name[38:-16]
    
    # acquisition date
    year_img = zip_name[11:15]
    month_img = zip_name[15:17]
    day_img = zip_name[17:19]
    discriminator_img = zip_name[-6:]
    
    # predicted files names
    search_criteria = "*_predicted.tif"
    q = os.path.join(class_path, search_criteria)
    dem_fps = glob.glob(q)
    dem_fps.sort(key=os.path.getmtime, reverse=True)
    # dem_fps.sort(key=os.path.getmtime, reverse=False)
    
    #dem_fps.sort(key=os.path.getmtime) ???? ai pega sempre o primeiro que é dia 18 afff

    
    for fname in dem_fps:
        print("Teste: ",fname)
        if grid_img in fname:

            # test date
            name_file = fname[-74:-14] # get the name of the file
            year_test = name_file[11:15]
            month_test = name_file[15:17]
            day_test = name_file[17:19]
            discriminator_test = name_file[-6:]

            path_shapes = poly_path + '/' + str(year_img) + str(month_img) + str(day_img) + 'T' + str(discriminator_img) + '_' + str(year_test) + str(month_test) + str(day_test) + 'T' + str(discriminator_test) + '_' + str(grid_img)

            #------------------------
            
            # d1 = datetime.datetime(int(year_img), int(month_img), int(day_img)) # current image
            # d2 = datetime.datetime(int(year_test), int(month_test), int(day_test)) # test image
            
            # print(d1)
            # print(d2)
            
            # days_dif = d1 - d2
            # days_d = abs(days_dif.days)
                
            # if time_spaced is None:
            #     time_spaced = days_d
            #     #time_spaced = xxx
            
            # if days_d > time_spaced:
            #     print(f"Images spaced more than {time_spaced} days... Getting the next image.")
            #     continue
            
            # elif d1 < d2:
            #     print("Previous image found for grid:", grid_img)
            #     print("Date of current image:", day_img, "-", month_img, '-', year_img)
            #     print("Date of previous image:", day_test, "-", month_test, '-', year_test)
            #     break
            #------------------------------------
            
            if os.path.isdir(path_shapes) is False:
                    
                # compare date - date in yyyy/mm/dd format 
                d1 = datetime.datetime(int(year_img), int(month_img), int(day_img)) # current image
                d2 = datetime.datetime(int(year_test), int(month_test), int(day_test)) # test image
                
                print(d1)
                print(d2)
                
                days_dif = d1 - d2
                days_d = abs(days_dif.days)
                
                if time_spaced is None:
                    time_spaced = days_d
                    #time_spaced = xxx
                    
                time_spaced = days_d # ADICIONEI PARA TESTAR
                
                if days_d > time_spaced:
                    print(f"Images spaced more than {time_spaced} days... Getting the next image.")
                    continue
            
                            
                elif d1 < d2: # because the download is made from the most recent image to the oldest 
                    print("Previous image found for grid:", grid_img)
                    print("Date of current image:", day_img, "-", month_img, '-', year_img)
                    print("Date of previous image:", day_test, "-", month_test, '-', year_test)
                                    
                    BSCL1 = cloud_masks(path_to_folder, save_class_path) # current
                    EVI1 = evi(path_to_folder)
                    EVI1[BSCL1 == 9999] = np.nan # masking EVI final image
                    
                    # search for the path
                    path_to_folder_BSCL2  = save_imgs + '/' + name_file + '.SAFE/GRANULE/'
                    BSCL2 = cloud_masks(path_to_folder_BSCL2, save_class_path) # old image
                    EVI0 = evi(path_to_folder_BSCL2)
                    EVI0[BSCL2 == 9999] = np.nan # masking EVI inital image
                    
                    #mask_BSCL = BSCL2 - BSCL1
                    #BSCL2[mask_BSCL == -9999] = 9999 # use BSCL2 as reference
                                
                    # create polygons
                    actual_open = rasterio.open(class_path + '/' + zip_name + '_predicted.tif')
                    old_open = rasterio.open(class_path + '/' + name_file + '_predicted.tif')
                    
                    actual = actual_open.read()
                    old = old_open.read()
                    
                    difference = old - actual
                    
                    difference[difference == 255] = 1 # deforestation (polygon is generated)
                    difference[difference == -255] = 0 # forest growth (polygon is not generated)
                    
                    # insert masks that represents clouds
                    difference[BSCL2 == 9999] = 0 # where there is clouds, the polygon is not generated
                    difference[BSCL1 == 9999] = 0
                    
                    # opening clouds masks from UNet
                    clouds_actual = rasterio.open(class_clouds + '/' + zip_name + '_predicted.tif')
                    clouds_old = rasterio.open(class_clouds + '/' + name_file + '_predicted.tif')
                    
                    clouds_actual_raster = clouds_actual.read()
                    clouds_old_raster = clouds_old.read()
                    
                    difference[clouds_actual_raster == 255] = 0 # where there is clouds, the polygon is not generated
                    difference[clouds_old_raster == 255] = 0
                    
                    # calling growth_rate() function
                    dt = d1 - d2
                    dt = dt.days # getting time interval between the two images in days
                    
                    #-----------------------------------------------------------------------------   
                    print('Applying growth rate function')
                    r_matrix, r_mask = growth_rate(EVI0[0,:,:],EVI1[0,:,:],dt,1)
    
                    arr_diff = difference[0,:,:]
                    arr_old = old[0,:,:]
    
                    r_mask[(arr_diff==0) & (arr_old==0)] = 0 # not forested areas in both comparisions
                    r_mask[arr_diff == -255] = 0 # areas where there weren't forests but now there are
                    
                    # excluding noisy pixels
                    img_r = copy.copy(r_mask)
                    kernel = np.ones((2,2),np.uint8)
                    erosion_r = cv2.erode(img_r,kernel,iterations = 1)
                    
                    erosion_r = np.int8(erosion_r)
                    
                    erosion_r = np.expand_dims(erosion_r, axis=0) # expanding channels to: [:,:,:]
                    
                    
                    #-----------------------------------------------------------------------------   
                    print('Applying clouds correction')
                    # Applying clouds corrections
                    clouds_defor = clouds_actual_raster + clouds_old_raster + difference
                    clouds_defor[clouds_defor > 0] = 1 
                    clouds_sum = clouds_actual_raster + clouds_old_raster
                    clouds_sum[clouds_sum > 1] = 1
                    
                    lw, num = measurements.label(clouds_defor[0,:,:])
                    
                    sum_all = clouds_sum[0,:,:] + lw # lw is the groups
                    
                    clouds_inter = np.zeros((lw.shape[0],lw.shape[1])) # tem as nuvens e o raster de deforestation
    
                    clouds_inter = np.where((lw!=sum_all),lw,0)
                    
                    clouds_inter[clouds_inter!=0]
                    uniques = np.unique(clouds_inter)
                    
                    teste_list = np.array(uniques).tolist() # 23690 classes em forma de lista
                    teste_list[0] = 1 # tirar o 0 como grupo
                    
                    lw_list = np.array(lw).tolist() # mapa com todas as classes em forma de lista
                    
                    values = np.ones((len(teste_list))).tolist() 
                    keys = teste_list # 23690 classes em forma de lista
    
                    # Changing the values that have clouds interference
                    arr = np.concatenate(lw_list)
                    idx = npi.indices(keys, arr, missing='mask')
                    remap = np.logical_not(idx.mask)
                    arr[remap] = np.array(values)[idx[remap]]
                    replaced = np.array_split(arr, np.cumsum([len(a) for a in lw_list][:-1]))
                    
                    final = np.array(replaced)
    
                    final[final != 1] = 0 # o que nao foram substituidos recebem 0
                    
                    # mask in defor variable (onde final for 1, defor recebe 0)
                    difference = difference[0,:,:]
                    difference[final == 1] = 0
                    
                    # excluding noisy pixels
                    img = copy.copy(difference)
                    kernel = np.ones((5,5),np.uint8)
                    erosion = cv2.erode(img,kernel,iterations = 1)
                    
                    erosion = np.expand_dims(erosion, axis=0) # expanding channels to: [:,:,:]
                    
                    #-----------------------------------------------------------------------------   
                    # selecting the polygons which have some pixel with negative growth
                    print('Evaluating falses positives')
                    lw, num = measurements.label(erosion[0,:,:]) #group pixels zones
                    
                    sum_all = erosion_r + lw
                    
                    defor_mask = np.zeros((lw.shape[0],lw.shape[1]))
                    
                    defor_mask = np.where((lw!=sum_all),lw,0)
                    
                    defor_mask[defor_mask!=0]
                    uniques = np.unique(defor_mask)
                    
                    teste_list = np.array(uniques).tolist()
                    teste_list[0] = 1 # tirar o 0 como grupo
                    
                    lw_list = np.array(lw).tolist() # mapa com todas as classes em forma de lista
                    values = np.ones((len(teste_list))).tolist() 
                    keys = teste_list 
                    dictionary = dict(zip(keys, values)) 
                    
                    arr = np.concatenate(lw_list)
                    idx = npi.indices(keys, arr, missing='mask')
                    remap = np.logical_not(idx.mask)
                    arr[remap] = np.array(values)[idx[remap]]
                    replaced = np.array_split(arr, np.cumsum([len(a) for a in lw_list][:-1]))
                    
                    final = np.array(replaced)
                    
                    final[final!=1]=0 # o que nao foram substituidos recebem 0
                    
                    np.shape(final)         
                    
                    final = np.expand_dims(final, axis=0)
    
                    erosion[final==0]=0
                    #-----------------------------------------------------------------------------                
                    # remove groups of pixels with an area smaller than 10 pixels
                    lw, num = measurements.label(erosion[0,:,:]) # grouping pixels clusters
                    area = measurements.sum(erosion[0,:,:], lw, index=arange(lw.max() + 1))
                    areaImg = area[lw]
    
                    areaImg = np.expand_dims(areaImg, axis=0)
                    
                    erosion[areaImg < 10] = 0 # IT CAN BE TESTED DIFFERENT VALUES TO CHOOSE THE BEST ONE
    
                    #-----------------------------------------------------------------------------                
                    # crs from a satellite image band - to save image metadata
                    listOfFiles = list()
                    
                    for (dirpath, dirnames, filenames) in os.walk(path_to_folder):
                        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
                        
                    for bandname in listOfFiles:
                        if (bandname.endswith("B02_10m.jp2")):
                            # read metadata information
                            b2 = rasterio.open(bandname)
                            
                    print('Creating shapefile')
                    # creating vectors of deforestation raster (erosion)
                    
                    shapes = rasterio.features.shapes(erosion, transform=b2.transform)
        
                    records = [{"geometry": geometry, "properties": {"value": value}}
                               for (geometry, value) in shapes if value == 1]
                    
                    schema = {"geometry": "Polygon", "properties": {"value": "int"}}
                    
                    os.mkdir(path_shapes)
                    with fiona.open(path_shapes + '/' + str(year_img) + str(month_img) + str(day_img) + '_' + str(year_test) + str(month_test) + str(day_test) + str(grid_img) + '.shp',
                                    "w", "ESRI Shapefile",
                                    crs=b2.crs.data, schema=schema) as out_file:
                        out_file.writerecords(records)
                        
                    #-------------------------------------------------                
    
                    print('Saving results')
                    with rasterio.open(poly_path + '/' + str(year_img) + str(month_img) + str(day_img) + '_' + str(year_test) + str(month_test) + str(day_test) + str(grid_img) + '_deforestation.tif',
                                       'w',driver='Gtiff', 
                                        width=b2.width, height=b2.height, 
                                        count=1,crs=b2.crs,transform=b2.transform,
                                        dtype='uint8') as deforestation:
                        deforestation.write(erosion) 
                        deforestation.close()
                        
                    with rasterio.open(poly_path + '/' + str(year_img) + str(month_img) + str(day_img) + '_' + str(year_test) + str(month_test) + str(day_test) + str(grid_img) + '_negative_growth.tif',
                                       'w',driver='Gtiff', 
                                        width=b2.width, height=b2.height, 
                                        count=1,crs=b2.crs,transform=b2.transform,
                                        dtype='int8') as rmask:
                        rmask.write(erosion_r) 
                        rmask.close()
                        
                        break # sai do laço for?
   
        
def def_main(save_imgs, save_rgb, save_tiles, unet_weights, unet_clouds,
             class_path, class_clouds, poly_path, 
             percentiles_forest, percentiles_clouds, geojson_file=None):
    
    n = int(input('Have you already downloaded the images? (0: NO, 1: YES): '))
    
    if n == 0:
        # download images
        download_images(save_imgs, save_rgb, save_tiles, unet_weights, unet_clouds,
                        class_path, class_clouds, poly_path, percentiles_forest, 
                        percentiles_clouds, geojson_file)
        
    elif n == 1:
        
        time_spaced = int(input('What is the time spacing between images, in days, that should be considered? (Sentinel-2 default is 5 days): '))
        
        app_images_test(save_imgs, save_rgb, save_tiles, unet_weights, unet_clouds,
                        class_path, class_clouds, poly_path, time_spaced,
                        percentiles_forest, percentiles_clouds, geojson_file=None)
    
    return
