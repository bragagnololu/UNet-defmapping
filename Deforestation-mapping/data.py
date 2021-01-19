from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
import rasterio

#from skimage import img_as_ubyte #adicionei

from PIL import Image
import numpy as np
import skimage

#Sky = [128,128,128]
#Building = [128,0,0]
#Pole = [192,192,128]
#Road = [128,64,128]
#Pavement = [60,40,222]
#Tree = [128,128,0]
#SignSymbol = [192,128,128]
#Fence = [64,64,128]
#Car = [64,0,128]
#Pedestrian = [64,64,0]
#Bicyclist = [0,128,192]
#Unlabelled = [0,0,0]

#COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
 #                         Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        #print(img.shape,img_elev.shape,img_slope.shape)
        
        #print(img.shape)
        img = img / 255
        #img_elev = img_elev / 255
        #img_slope = img_slope / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        #print(img.size)
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (512,512),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    #VER SE NAO DÁ PROBLEMA NOS VALORES DOS RASTERS! -> TESTAR SALVANDO IMAGEM
#    image_generator_elev = image_datagen.flow_from_directory(
#        train_path,
#        classes = ['elev_png2'],
#        class_mode = None,
#        color_mode = "grayscale",
#        target_size = target_size,
#        batch_size = batch_size,
#        #save_to_dir = save_to_dir,
#        save_to_dir = save_to_dir,        
#        save_prefix  = image_save_prefix,
#        seed = seed)
#    
#    image_generator_slope = image_datagen.flow_from_directory(
#        train_path,
#        classes = ['slope_png2'],
#        class_mode = None,
#        color_mode = "grayscale",
#        target_size = target_size,
#        batch_size = batch_size,
#        save_to_dir = save_to_dir,
#        save_prefix  = image_save_prefix,
#        seed = seed)
    
    #Mask generator
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img,mask) in train_generator: 
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        #print(img_elev)
        yield (img, mask)
        #yield (img,mask)
        
#        img = np.dstack((img[0],img_elev[0],img_slope[0]))
#        img = img.reshape((-1, 256, 256, 5))
#        mask = mask[0] #BLREEEEEEEEEEEEE
#        #AQUI DARIA PRA JUNTAR TODAS AS CAMADAS COM O DSTACK!!!
        #print("oi1")
#    while True:
#        X1i = image_generator.next()
#        X2i = image_generator_elev.next()
#        X3i = image_generator_slope.next()
#        X4i = mask_generator.next()  
#    
#        imgFirst = X1i[0]
#        imgSecond = X2i[0]
#        imgThird = X3i[0]
#        imgMask = X4i[0]
#        imgi=np.zeros((len(imgFirst), 256, 256, 5), dtype=np.float32)
#        for n in range(0,len(imgFirst)):
#            img1 = imgFirst
#            img2 = imgSecond
#            img3 = imgThird
#
#            imgi[n] = np.dstack((img1,img2,img3))
#        #imgi = imgi.reshape((-1, 256, 256, 5))
#
#            
#        imgi = imgi / 255
#        imgMask = imgMask /255
#        imgMask[imgMask > 0.5] = 1
#        imgMask[imgMask <= 0.5] = 0
#            
#        yield imgi, imgMask  #Yield both images and their mutual label   
#    
    
        

def valGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (512,512),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        shuffle = False,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        #seed = seed
        )
    
    #VER SE NAO DÁ PROBLEMA NOS VALORES DOS RASTERS! -> TESTAR SALVANDO IMAGEM
#    image_generator_elev = image_datagen.flow_from_directory(
#        train_path,
#        classes =  ['elev_png2'],
#        class_mode = None,
#        color_mode = "grayscale",
#        target_size = target_size,
#        batch_size = batch_size,
#        save_to_dir = save_to_dir,
#        save_prefix  = image_save_prefix,
#        seed = seed)
#    
#    image_generator_slope = image_datagen.flow_from_directory(
#        train_path,
#        classes =  ['slope_png2'],
#        class_mode = None,
#        color_mode = "grayscale",
#        target_size = target_size,
#        batch_size = batch_size,
#        save_to_dir = save_to_dir,
#        save_prefix  = image_save_prefix,
#        seed = seed)
    
    
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        shuffle = False,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        #seed = seed
        )
    
    val_generator = zip(image_generator, mask_generator)
#    for (img,img_elev,img_slope,mask) in train_generator:
#        img,mask = adjustData(img,img_elev,img_slope,mask,flag_multi_class,num_class)
#        #AQUI DARIA PRA JUNTAR TODAS AS CAMADAS COM O DSTACK!!!
#    yield (img,mask)
    for (img,mask) in val_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        #print(img.shape)
        yield (img, mask)
#    while True:
#        X1i = image_generator.next()
#        X2i = image_generator_elev.next()
#        X3i = image_generator_slope.next()
#        X4i = mask_generator.next()            
#
#        imgFirst = X1i[0]
#        imgSecond = X2i[0]
#        imgThird = X3i[0]
#        imgMask = X4i[0]
#        imgi=np.zeros((len(imgFirst), 256, 256, 5), dtype=np.float32)
#        for n in range(0,len(imgFirst)):
#            img1 = imgFirst[n]
#            img2 = imgSecond[n]
#            img3 = imgThird[n]
#            imgi[n] = np.dstack((img1,img2,img3))
#
#            
#        imgi = imgi / 255
#        imgMask = imgMask /255
#        imgMask[imgMask > 0.5] = 1
#        imgMask[imgMask <= 0.5] = 0
#                
#        yield imgi, imgMask  #Yield both images and their mutual label  
#    while True:
#        X1i = image_generator.next()
#        X2i = image_generator_elev.next()
#        X3i = image_generator_slope.next()
#        X4i = mask_generator.next()  
#    
#        imgFirst = X1i[0]
#        imgSecond = X2i[0]
#        imgThird = X3i[0]
#        imgMask = X4i[0]
#        imgi=np.zeros((len(imgFirst), 256, 256, 5), dtype=np.float32)
#        for n in range(0,len(imgFirst)):
#            img1 = imgFirst
#            img2 = imgSecond
#            img3 = imgThird
#
#            imgi[n] = np.dstack((img1,img2,img3))
#        #imgi = imgi.reshape((-1, 256, 256, 5))
#
#            
#        imgi = imgi / 255
#        imgMask = imgMask /255
#        imgMask[imgMask > 0.5] = 1
#        imgMask[imgMask <= 0.5] = 0
#            
#        yield imgi, imgMask  #Yield both images and their mutual label  


def testGenerator(test_path,percentiles,clouds = 0,target_size = (512,512),flag_multi_class = False,as_gray = False): # inserir como entrada os percentis
# se avaliar clouds = 1
    
    listOfFiles = list()
    
    for (dirpath, dirnames, filenames) in os.walk(test_path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    
    for fname in listOfFiles:
        
        raster = rasterio.open(fname,as_gray = as_gray)
        img = raster.read()
        
        lines = np.size(img,1)
        cols = np.size(img,2)
        
        if lines != 512 & cols == 512:
            n = 512 - lines
            npad = ((0, 0), (n,0), (0, 0)) #0 adição na dimensão k
            img = np.pad(img, pad_width=npad, mode='constant', constant_values=255)
            img = np.transpose(img, (1, 2, 0)) #channels last
            
        elif lines == 512 & cols != 512:
            n = 512 - cols
            npad = ((0, 0), (0,0), (0, n)) #0 adição na dimensão k
            img = np.pad(img, pad_width=npad, mode='constant', constant_values=255)
            img = np.transpose(img, (1, 2, 0)) #channels last
            
        elif lines != 512 & cols != 512:
            m = 512 - lines
            n = 512 - cols
            npad = ((0, 0), (m,0), (0, n))
            img = np.pad(img, pad_width=npad, mode='constant', constant_values=255)
            img = np.transpose(img, (1, 2, 0)) #channels last
            
        else:
            img = np.transpose(img, (1, 2, 0)) #channels last
            

        img[img > 10000] = 10000
        img = img.astype(float)/10000
        
        bands_third = np.load(percentiles[0])
        bands_nin = np.load(percentiles[1])
        
        if clouds == 1:
            img = img[:,:,0:3]
        
        for i in range(len(bands_third)):
            img[:,:,i] = (img[:,:,i] - bands_third[i])/(bands_nin[i] - bands_third[i])
        #img = trans.resize(img,target_size)
        #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img
        

def saveResult(tiles_dir,save_path,npyfile,flag_multi_class = False,num_class = 2):

    listOfFiles = list()
    listOfFiles_png = list()

    for (dirpath, dirnames, filenames) in os.walk(tiles_dir):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    listOfFiles_png += [file.replace(".tif",".png") for file in filenames]
    
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        
        
        img[img > 0.5] = 1 #ADICIONEI
        img[img <= 0.5] = 0 #ADICIONEI
        
        img = skimage.util.img_as_ubyte(img) #0 a 255
        
        #Open a reference image
        timg = rasterio.open(listOfFiles[i])
        
        lines = np.size(timg,0)
        cols = np.size(timg,1)
        
        if lines != 512 & cols == 512:
            n = 512 - lines
            dell = slice(0,n)
            img = np.delete(img,dell,axis=0)
            img = np.expand_dims(img, axis=0) #channels first
            
        elif lines == 512 & cols != 512:
            n = 512 - cols
            del1 = slice(512-n,512)
            img = np.delete(img,del1,axis=1)
            img = np.expand_dims(img, axis=0) #channels first
            
        elif lines != 512 & cols != 512:
            m = 512 - lines
            n = 512 - cols
            del1 = slice(0,m)
            del2 = slice(512-n,512)
            img = np.delete(img,del1,axis=0)
            img = np.delete(img,del2,axis=1)
            img = np.expand_dims(img, axis=0) #channels first
            
        else:
            img = np.expand_dims(img, axis=0) #channels first
        
        with rasterio.open(os.path.join(save_path,listOfFiles_png[i]),'w',driver='Gtiff', width=timg.width, height=timg.height, 
                count=1,crs=timg.crs,transform=timg.transform, dtype='uint8') as predicted_img: # MUDAR DPS PARA UINT8
            #Provavelmente vai dar algum problema aqui pq a RGB contem 3 canais e aqui só 1
            predicted_img.write(img) 
            predicted_img.close()
        
        #io.imsave(os.path.join(save_path,"%d_predict.png"%i),skimage.util.img_as_ubyte(img))


def testGenerator3(test_path,num_image = 70,target_size = (512,512),flag_multi_class = False,as_gray = False):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.tiff"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def testGenerator2(test_path,num_image = 70,target_size = (512,512),flag_multi_class = False,as_gray = False):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.tiff"%i),as_gray = as_gray)
        img_elev = io.imread(os.path.join("D:\\Usuarios\\Xeon\\Desktop\\geo_database_landslide\\teste\\elev_png2","%d.png"%i),as_gray = True)
        img_slope = io.imread(os.path.join("D:\\Usuarios\\Xeon\\Desktop\\geo_database_landslide\\teste\\slope_png2","%d.png"%i),as_gray = True)
        
        img = img / 255
        img_elev = img_elev / 255
        img_slope = img_slope / 255

        img = trans.resize(img,target_size)
        img_elev = trans.resize(img_elev,target_size)
        img_slope = trans.resize(img_slope,target_size)
        
#        print(img.shape)
#        print(img_elev.shape)
#        print(img_slope.shape)
        
        img_all = np.dstack((img, img_elev, img_slope))
        #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img_all = np.reshape(img_all,(1,)+img_all.shape)
        yield img_all

def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255